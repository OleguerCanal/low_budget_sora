import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from low_budget_sora import DEBUG_GIF_PATH
from low_budget_sora.model.diffusion_transformer import DiffusionTransformer
from low_budget_sora.model.noise_scheduler import NoiseScheduler
from low_budget_sora.data.utils import console_print, tensor_video_to_gif

class DiffusionTransformerLitModule(L.LightningModule):
    def __init__(
        self,
        model: DiffusionTransformer,
        noise_schedule: NoiseScheduler,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "noise_schedule"])
        self.model = model
        self.noise_schedule = noise_schedule
        self.lr = lr

    def _forward_diffusion(self, x0: torch.Tensor):
        """
        x0: (B, T, C, H, W) in [0,1]
        returns: x_t, eps, t
        """
        B = x0.size(0)
        device = x0.device

        t = torch.randint(
            low=0,
            high=self.noise_schedule.num_steps,
            size=(B,),
            device=device,
            dtype=torch.long,
        )
        eps = torch.randn_like(x0)
        x_t = self.noise_schedule.q_sample(x0, t, noise=eps)
        return x_t, eps, t

    def training_step(self, batch, batch_idx):
        x_0 = batch["video"]        # (B, T, 1, H, W), float in [0,1]
        prompt_ids = batch["prompt_ids"]

        x_t, eps, t = self._forward_diffusion(x_0)
        eps_hat = self.model.forward(
            x_t=x_t,
            diffusion_timestep=t,
            prompt_ids=prompt_ids,
        )

        loss = F.mse_loss(eps_hat, eps)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0 = batch["video"]
        prompt_ids = batch["prompt_ids"]

        x_t, eps, t = self._forward_diffusion(x0)
        eps_hat = self.model.forward(
            x_t=x_t,
            diffusion_timestep=t,
            prompt_ids=prompt_ids,
        )

        loss = F.mse_loss(eps_hat, eps)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
