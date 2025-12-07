import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import make_grid
from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.util import timestep_embedding

class LatentFlowMatching(pl.LightningModule):
    def __init__(self,
                 dit_config,
                 first_stage_config,
                 cond_stage_config=None, # Not used for mask concat but kept for compatibility
                 learning_rate=1e-4,
                 monitor="val/loss",
                 latent_size=32,
                 channels=3,
                 mask_channels=1
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['first_stage_config', 'cond_stage_config'])
        
        # 1. Load First Stage Model (VQ-VAE or Autoencoder)
        self.first_stage_model = instantiate_from_config(first_stage_config)
        self.first_stage_model.eval()
        self.first_stage_model.train = False
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
            
        # 2. Initialize DiT Model
        self.model = instantiate_from_config(dit_config)
        
        self.learning_rate = learning_rate
        self.monitor = monitor
        self.latent_size = latent_size
        self.channels = channels
        self.mask_channels = mask_channels

    def get_input(self, batch, k):
        # Image
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        
        # Segmentation Mask (Condition)
        mask = batch["segmentation"]
        if len(mask.shape) == 3:
            mask = mask[..., None]
        mask = mask.permute(0, 3, 1, 2).float()
        
        # Resize mask to latent size directly using nearest to keep binary nature if needed
        # Assuming Latent is downsampled by factor f (e.g., 4 or 8)
        # Here we resize strictly to self.latent_size
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
        # 1. Encode Image to Latent space (z_0: Data)
        with torch.no_grad():
            encoder_posterior = self.encode_first_stage(x)
            z_0 = self.get_first_stage_encoding(encoder_posterior).detach() # Data
        
        # 2. Sample Noise (z_1: Noise)
        z_1 = torch.randn_like(z_0)
        
        # 3. Sample Time t ~ Uniform[0, 1]
        bs = z_0.shape[0]
        t = torch.rand((bs,), device=self.device)
        
        # 4. Flow Matching Interpolation (Optimal Transport Path)
        # z_t = (1 - t) * z_0 + t * z_1
        # t needs broadcasting
        t_b = t.view(bs, 1, 1, 1)
        z_t = (1 - t_b) * z_0 + t_b * z_1
        
        # 5. Target Vector Field u_t
        # d/dt (z_t) = z_1 - z_0
        target_v = z_1 - z_0
        
        # 6. Predict Vector Field
        # Timestep embedding needed for DiT
        # We produce sinusoidal embedding of dimension hidden_size of DiT
        t_emb = timestep_embedding(t * 1000, self.model.t_embedder[0].in_features, repeat_only=False).to(self.device)
        
        pred_v = self.model(z_t, t_emb, mask)
        
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
        
        # Log inputs
        log["inputs"] = x
        log["condition_mask"] = mask.repeat(1,3,1,1) # Make 3 channels for visualization
        
        # Sampling (Euler Method for Flow Matching)
        # z_0 = z_1 + integral(v_pred dt) from 1 to 0
        # Wait, our formulation is z_t = (1-t)z_0 + t z_1
        # t goes from 0 (data) to 1 (noise).
        # Generative process: t goes from 1 (noise) to 0 (data).
        # ODE: dz/dt = v_pred(z, t).
        # Discretization: z_{t-dt} = z_t - v_pred(z_t, t) * dt
        
        steps = 50
        dt = 1.0 / steps
        z = torch.randn(N, self.channels, self.latent_size, self.latent_size, device=self.device) # Start from noise (t=1)
        
        for i in range(steps):
            t_val = 1.0 - i * dt # t goes 1.0 -> 0.0
            t_batch = torch.full((N,), t_val, device=self.device)
            t_emb = timestep_embedding(t_batch * 1000, self.model.t_embedder[0].in_features, repeat_only=False).to(self.device)
            
            v_pred = self.model(z, t_emb, mask)
            
            # Euler step: z_{next} = z_{curr} - v * dt
            z = z - v_pred * dt
            
        # Decode Latents
        x_rec = self.first_stage_model.decode(z)
        log["samples"] = x_rec
        
        return log