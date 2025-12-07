import os
import glob
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import albumentations
from torch.utils.data import Dataset


# ----- 마스크 이진화(필요 시 임계값 조정) -----
def preprocess_mask(img: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(img)
    mask[img >= 150] = 1
    return mask


# ----- 유효 이미지/마스크 확장자 -----
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _is_valid_image_name(name: str) -> bool:
    """Thumbs.db, .DS_Store 등과 이미지 외 확장자 필터링"""
    name_l = name.strip().lower()
    if not name_l:
        return False
    if name_l in ("thumbs.db", ".ds_store"):
        return False
    return Path(name_l).suffix.lower() in IMG_EXTS


# ----- 마스크 확장자 탐색 유틸 -----
def _resolve_mask_path(seg_root: str, img_name_or_path: str) -> str:
    """
    이미지 파일명(또는 경로)에서 stem을 가져와 seg_root에서
    .png/.jpg/.jpeg/.bmp 순서로 존재하는 마스크를 찾아 반환.
    """
    p = Path(img_name_or_path)
    stem = p.stem  # 예: N2_4032873_2_01
    seg_dir = Path(seg_root)
    for ext in MASK_EXTS:
        cand = seg_dir / f"{stem}{ext}"
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        f"Mask not found for '{stem}' in '{seg_dir}' (tried {MASK_EXTS})"
    )


# =========================
# Dataset 구현
# =========================
class SegmentationBase(Dataset):
    def __init__(
        self,
        data_csv,
        data_root,
        segmentation_root,
        size=None,
        random_crop=False,
        interpolation="bicubic",
        n_labels=2,
        shift_segmentation=False,
    ):
        self.n_labels = n_labels
        self.shift_segmentation = shift_segmentation
        self.data_csv = data_csv
        self.data_root = data_root
        self.segmentation_root = segmentation_root

        # --- CSV 읽고 유효 이미지명만 필터링 ---
        with open(self.data_csv, "r") as f:
            raw_lines = [ln.strip() for ln in f.read().splitlines()]
        filtered = [ln for ln in raw_lines if _is_valid_image_name(ln)]

        file_paths, seg_paths = [], []
        skipped = 0
        for l in filtered:
            img_path = os.path.join(self.data_root, l)
            if not os.path.exists(img_path):
                skipped += 1
                continue
            # 마스크 경로 해석 (없으면 스킵)
            try:
                seg_path = _resolve_mask_path(self.segmentation_root, l)
            except FileNotFoundError:
                try:
                    seg_path = _resolve_mask_path(self.segmentation_root, img_path)
                except FileNotFoundError:
                    skipped += 1
                    continue

            file_paths.append(img_path)
            seg_paths.append(seg_path)

        # 최종 목록 반영
        self.image_paths = [Path(p).name for p in file_paths]
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [Path(p).name for p in file_paths],
            "file_path_": file_paths,
            "segmentation_path_": seg_paths,
        }

        if skipped:
            print(
                f"[kvasir] filtered out {skipped} invalid/missing-mask entries "
                f"(from {len(raw_lines)} lines in {self.data_csv})."
            )

        # --- 리사이즈/크롭 설정 ---
        size = None if size is not None and size <= 0 else size
        self.size = size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4,
            }[self.interpolation]

            self.image_rescaler = albumentations.SmallestMaxSize(
                max_size=self.size, interpolation=self.interpolation
            )
            self.segmentation_rescaler = albumentations.SmallestMaxSize(
                max_size=self.size, interpolation=cv2.INTER_NEAREST
            )

            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = self.cropper

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        # ----- 이미지 -----
        image = Image.open(example["file_path_"])
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]

        # ----- 마스크 (경로 보강) -----
        seg_path = example["segmentation_path_"]
        if not os.path.exists(seg_path):
            seg_path = _resolve_mask_path(
                self.segmentation_root, example["relative_file_path_"]
            )
            example["segmentation_path_"] = seg_path

        segmentation = Image.open(seg_path)
        if segmentation.mode != "L":
            segmentation = segmentation.convert("L")
        segmentation = np.array(segmentation).astype(np.uint8)

        # 이진화 등 전처리
        segmentation = preprocess_mask(segmentation)

        if self.shift_segmentation:
            # unlabeled==255 같은 경우 +1 offset
            segmentation = segmentation + 1

        if self.size is not None:
            segmentation = self.segmentation_rescaler(image=segmentation)["image"]

        if self.size is not None:
            processed = self.preprocessor(image=image, mask=segmentation)
        else:
            processed = {"image": image, "mask": segmentation}

        example["image"] = (processed["image"] / 127.5 - 1.0).astype(np.float32)
        
        # onehot encoding
        # segmentation = np.squeeze(segmentation, axis=-1)
        # onehot = np.eye(self.n_labels)[segmentation]
        example["segmentation"] = processed["mask"][..., None].astype(np.float32)
        return example


# 예제용(원본 유지)
class Examples(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(
            data_csv="data/sflckr_examples.txt",
            data_root="data/sflckr_images",
            segmentation_root="data/sflckr_segmentations",
            size=size,
            random_crop=random_crop,
            interpolation=interpolation,
        )


# Kvasir train/eval (경로만 환경에 맞게)
class KvasirSegTrain(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(
            data_csv='../../../01_data/02_preproc/02_abnormal/P2/data_train.txt',
            data_root='../../../01_data/02_preproc/02_abnormal/P2/images',
            segmentation_root='../../../01_data/02_preproc/02_abnormal/P2/masks',
            size=size,
            random_crop=random_crop,
            interpolation=interpolation,
            n_labels=2,
        )


class KvasirSegEval(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(
            data_csv='../../../01_data/02_preproc/02_abnormal/P2/data_eval.txt',
            data_root='../../../01_data/02_preproc/02_abnormal/P2/images',
            segmentation_root='../../../01_data/02_preproc/02_abnormal/P2/masks',
            size=size,
            random_crop=random_crop,
            interpolation=interpolation,
            n_labels=2,
        )


# ===== CSV 생성 유틸 (원본 유지) =====
def write_lines(file, lines):
    with open(file, "w") as f:
        for line in lines:
            f.write(os.path.basename(line))
            f.write("\n")


def generateKvasirCSV(dir, output, train=0.9):
    files = glob.glob(dir)
    random.shuffle(files)
    length = len(files)

    train_data = files[: int(train * length)]
    write_lines(f"{output}/kvasir_train.txt", train_data)

    eval_data = files[int(train * length) :]
    write_lines(f"{output}/kvasir_eval.txt", eval_data)
