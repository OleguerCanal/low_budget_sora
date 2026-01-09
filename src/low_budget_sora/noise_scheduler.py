import torch
import torch.nn as nn

class NoiseScheduler(nn.Module):
    def __init__(
            self, 
            num_steps: int, 
            beta_start=1e-4, 
            beta_end=0.02, 
            device="cpu",
        ):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, num_steps, device=device)  # (T,)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)                           # (T,)

        self.num_steps = num_steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        """
        x0:    (B, T, C, H, W) or any shape (B, ...)
        t:     (B,) LongTensor with values in [0, num_steps-1]
        noise: same shape as x0; if None, sampled from N(0,1)
        Returns x_t = sqrt(a_t) * x0 + sqrt(1-a_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # gather per-sample scalars and reshape for broadcasting
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x0.dim() - 1)))
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x0.dim() - 1)))

        return sqrt_ac * x0 + sqrt_om * noise
