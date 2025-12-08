import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from functools import partial
from torch.utils.checkpoint import checkpoint

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed

class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=32, patch_size=2, in_chans=4, embed_dim=768, bias=True):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, Embed_Dim, H/P, W/P) -> (B, Embed_Dim, N) -> (B, N, Embed_Dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim, pt_seq_len, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        # Create fixed frequency for RoPE
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x shape: [B, num_heads, N, head_dim]
        # We assume N corresponds to a square grid HxW
        # Standard 2D RoPE logic could be complex, here we apply 1D RoPE to the flattened sequence 
        # or implement 2D RoPE if spatial structure is critical.
        # For simplicity and compatibility with DiT (which flattens patches), we use 1D RoPE on the sequence.
        
        seq_len = x.shape[2]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq) # [seq_len, dim/2]
        
        # Create cos/sin tables [1, 1, seq_len, dim]
        freqs = torch.cat((freqs, freqs), dim=-1)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        
        return self.apply_rope(x, cos, sin)

    def apply_rope(self, x, cos, sin):
        # rotate_half
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)

def modulate(x, shift, scale):
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., 
                 norm_layer=nn.LayerNorm, fused_attn=True, use_rmsnorm=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x, rope=None):
        B, N, C = x.shape
        # (B, N, 3, Heads, Dim) -> (3, B, Heads, N, Dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if rope is not None:
            # rope is an instance of VisionRotaryEmbeddingFast, calling it applies embedding
            q = rope(q)
            k = rope(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LightningDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_qknorm=False, 
                 use_swiglu=False, use_rmsnorm=False, wo_shift=False, **block_kwargs):
        super().__init__()
        
        norm_layer = RMSNorm if use_rmsnorm else partial(nn.LayerNorm, elementwise_affine=False, eps=1e-6)
        
        self.norm1 = norm_layer(hidden_size)
        self.norm2 = norm_layer(hidden_size)
            
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm, norm_layer=norm_layer, **block_kwargs
        )
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
            
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True))
        else:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.wo_shift = wo_shift

    def forward(self, x, c, feat_rope=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            shift_msa, shift_mlp = None, None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        norm_layer = RMSNorm if use_rmsnorm else partial(nn.LayerNorm, elementwise_affine=False, eps=1e-6)
        self.norm_final = norm_layer(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LightningDiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=1,
        in_channels=4, # 3 (Latent) + 1 (Mask)
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False,
        use_rope=True, # Default True as requested
        use_rmsnorm=False,
        wo_shift=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels - 1 # Remove mask channel for output
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_checkpoint = use_checkpoint

        # 1. Patch Embedding (Handles Mask Concat Input)
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        
        # 2. Time Embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # 3. Position Embedding (Fixed Sin-Cos)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # 4. RoPE
        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = (input_size // patch_size) ** 2
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim * 2, # Full head dim passed to RoPE module logic
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None

        # 5. DiT Blocks
        self.blocks = nn.ModuleList([
            LightningDiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, 
                use_qknorm=use_qknorm, use_swiglu=use_swiglu, 
                use_rmsnorm=use_rmsnorm, wo_shift=wo_shift
            ) for _ in range(depth)
        ])
        
        # 6. Final Layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None: nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, mask):
        """
        x: (N, C_latent, H, W)
        t: (N,)
        mask: (N, C_mask, H, W)
        """
        # 1. Mask Conditioning via Concatenation
        x = torch.cat([x, mask], dim=1) # (N, 4, H, W)

        # 2. Patchify & Embed
        x = self.x_embedder(x) + self.pos_embed
        
        # 3. Time Embed (Conditioning)
        c = self.t_embedder(t) # (N, D) - No class label y used

        # 4. Transformer Blocks
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(block, x, c, self.feat_rope, use_reentrant=True)
            else:
                x = block(x, c, self.feat_rope)

        # 5. Finalize
        x = self.final_layer(x, c)
        x = self.unpatchify(x) # (N, 3, H, W)
        return x

# Pos Embedding Helpers
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)