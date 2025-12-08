import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torchvision.utils import make_grid
from ldm.util import instantiate_from_config

class LatentFlowMatching(pl.LightningModule):
    def __init__(self,
                 dit_config,
                 first_stage_config,
                 cond_stage_config=None, 
                 learning_rate=1e-4,
                 monitor="val/loss",
                 latent_size=32,
                 channels=3,
                 mask_channels=1
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['first_stage_config', 'cond_stage_config'])
        
        # 1. Load First Stage Model
        self.first_stage_model = instantiate_from_config(first_stage_config)
        self.first_stage_model.eval()
        self.first_stage_model.requires_grad_(False)
            
        # 2. Initialize DiT Model
        self.model = instantiate_from_config(dit_config)
        
        self.learning_rate = learning_rate
        self.monitor = monitor
        self.latent_size = latent_size
        self.channels = channels
        self.mask_channels = mask_channels

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        
        mask = batch["segmentation"]
        if len(mask.shape) == 3:
            mask = mask[..., None]
        mask = mask.permute(0, 3, 1, 2).float()
        
        # Resize mask to latent size
        mask = F.interpolate(mask, size=(self.latent_size, self.latent_size), mode="nearest")
        
        return x, mask

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    @torch.no_grad()
    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, torch.Tensor):
            return encoder_posterior
        return encoder_posterior.sample()

    def forward(self, x, mask):
        # 1. Encode Image (z_0: Data)
        with torch.no_grad():
            encoder_posterior = self.encode_first_stage(x)
            z_0 = self.get_first_stage_encoding(encoder_posterior).detach() 
        
        # 2. Sample Noise (z_1: Noise)
        z_1 = torch.randn_like(z_0)
        
        # 3. Sample Time t ~ Uniform[0, 1]
        bs = z_0.shape[0]
        t = torch.rand((bs,), device=self.device)
        
        # 4. Flow Matching Interpolation
        t_b = t.view(bs, 1, 1, 1)
        z_t = (1 - t_b) * z_0 + t_b * z_1
        
        # 5. Target Vector Field
        target_v = z_1 - z_0
        
        # 6. Predict Vector Field
        t_input = t * 1000 
        
        pred_v = self.model(z_t, t_input, mask)
        
        # 7. Loss
        loss = F.mse_loss(pred_v, target_v)
        
        return loss

    def training_step(self, batch, batch_idx):
        x, mask = self.get_input(batch, "image")
        loss = self(x, mask)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask = self.get_input(batch, "image")
        loss = self(x, mask)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    @torch.no_grad()
    def log_images(self, batch, N=4, **kwargs):
        log = dict()
        x, mask = self.get_input(batch, "image")
        x = x[:N]
        mask = mask[:N]
        
        log["inputs"] = x
        log["condition_mask"] = mask.repeat(1,3,1,1) 
        
        # Sampling (Euler Method)
        steps = 50
        dt = 1.0 / steps
        z = torch.randn(N, self.channels, self.latent_size, self.latent_size, device=self.device)
        
        for i in range(steps):
            t_val = 1.0 - i * dt 
            t_batch = torch.full((N,), t_val, device=self.device)

            t_input = t_batch * 1000
            
            v_pred = self.model(z, t_input, mask)
            
            z = z - v_pred * dt
            
        x_rec = self.first_stage_model.decode(z)
        log["samples"] = x_rec
        
        return log