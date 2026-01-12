import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from low_budget_sora import DEBUG_GIF_PATH
from low_budget_sora.model.diffusion_transformer import DiffusionTransformer
from low_budget_sora.model.noise_scheduler import NoiseScheduler
from low_budget_sora.data.utils import console_print, tensor_video_to_gif

local_rank = int(os.environ.get("LOCAL_RANK", -1))

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
        self.rng = None

    def _forward_diffusion(self, x0: torch.Tensor):
        """
        x0: (B, T, C, H, W) in [0,1]
        returns: x_t, eps, t
        """
        B = x0.size(0)
        device = x0.device
        
        if self.rng is None:
            self.rng = torch.Generator(device=device)
            self.rng.manual_seed(local_rank)
        
        t = torch.randint(
            low=0,
            high=self.noise_schedule.num_steps,
            size=(B,),
            device=device,
            dtype=torch.long,
            generator=self.rng,
        )
        eps = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, generator=self.rng)
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
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            logger=True,
            batch_size=x_0.size(0),
        )
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
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
            logger=True,
            batch_size=x0.size(0),
        )
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

    @torch.no_grad()
    def sample_videos(
            self,
            prompt_ids: torch.Tensor,
            num_steps: int,
            eta: float = 0.0,
        ) -> torch.Tensor:
        
        device = next(self.parameters()).device
        prompt_ids = prompt_ids.to(device)
        
        # Shapes
        video_patcher = self.model.video_patcher
        B = prompt_ids.size(0)
        T, C, H, W = video_patcher.T, video_patcher.C, video_patcher.H, video_patcher.W


        # Timesteps
        num_steps = self.noise_schedule.num_steps if num_steps is None else num_steps
        assert num_steps <= self.noise_schedule.num_steps
        timesteps = torch.linspace(
            self.noise_schedule.num_steps - 1, 0, num_steps, dtype=torch.long, device=device
        )

        # start from pure noise
        x_t = torch.randn(B, T, C, H, W, device=device)
        alphas_cumprod = self.noise_schedule.alphas_cumprod.to(device)  # (num_steps,)

        for i in range(num_steps):
            if i % 200 == 0:
                tensor_video_to_gif(x_t[0], f"generated_step_{i}.gif")
            t = timesteps[i]
            t_int = int(t.item())

            # current and previous alpha_bar
            alpha_t = alphas_cumprod[t_int]
            if t_int > 0:
                alpha_prev = alphas_cumprod[t_int - 1]
            else:
                alpha_prev = torch.tensor(1.0, device=device)

            # expand to broadcast over video tensor
            alpha_t_sqrt = alpha_t.sqrt()
            one_minus_alpha_t_sqrt = (1.0 - alpha_t).sqrt()

            # predict noise with the model
            t_batch = torch.full((B,), t_int, device=device, dtype=torch.long)
            eps_theta = self.model.forward(
                x_t=x_t,
                diffusion_timestep=t_batch,
                prompt_ids=prompt_ids,
            )  # (B, T, C, H, W)

            # x0 prediction
            x0_pred = (x_t - one_minus_alpha_t_sqrt * eps_theta) / alpha_t_sqrt

            if t_int == 0:
                x_t = x0_pred
                break

            # DDIM update
            alpha_prev_sqrt = alpha_prev.sqrt()
            sigma_t = (
                eta
                * torch.sqrt(
                    (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
                )
            )

            # dir term
            dir_coeff = torch.sqrt(torch.clamp(1 - alpha_prev - sigma_t**2, min=0.0))
            z = torch.randn_like(x_t) if eta > 0.0 else torch.zeros_like(x_t)

            x_t = alpha_prev_sqrt * x0_pred + dir_coeff * eps_theta + sigma_t * z

        # map back to [0,1] (we trained in [0,1] space)
        x_t = x_t.clamp(0.0, 1.0)
        return x_t