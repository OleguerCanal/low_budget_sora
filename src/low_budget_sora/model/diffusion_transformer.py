import math

import torch
import torch.nn as nn

from low_budget_sora.model.video_patcher import VideoPatcher


def sinusoidal_diffusion_timestep_embedding(
    diffusion_timestep: torch.Tensor,  # (B,)
    dim: int,  # (B, dim)
) -> torch.Tensor:
    """For each diffusion timestep, return a embedding of dimension `dim` using the sinusoidal embedding.
    """
    device = diffusion_timestep.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(1e4) * torch.arange(0, half, dtype=torch.float32, device=device) / half
    )
    args = diffusion_timestep.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb  # (B, dim)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.norm3 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x, cond, cond_mask = None):
        # x:    (B, N, d)
        # cond: (B, L, d)
        # cond_mask: (B, L) bool, True where pad (to ignore)

        # self-attention over video tokens
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]

        # cross-attention: video queries, prompt keys/values
        h = self.norm2(x)
        x = x + self.cross_attn(
            h, cond, cond,
            key_padding_mask=cond_mask,  # True -> ignore
            need_weights=False,
        )[0]

        # MLP
        h = self.norm3(x)
        x = x + self.mlp(h)
        return x


class DiffusionTransformer(nn.Module):
    def __init__(
            self,
            video_patcher: VideoPatcher,
            
            # Prompt
            prompt_vocab_size: int,
            max_prompt_len: int,
            
            # Model
            hidden_dim: int,
            mlp_ratio: float,
            num_heads: int,
            num_layers: int,
            
            pad_token_id: int = 0,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim

        self.video_patcher = video_patcher
        self.video_proj = nn.Linear(video_patcher.patch_dim, hidden_dim)
        
        # Prompt embedding
        self.token_embed = nn.Embedding(prompt_vocab_size, hidden_dim)
        self.pad_token_id = pad_token_id
        
        
        # learned absolute positional embeddings
        self.pos_embed_video = nn.Parameter(
            torch.zeros(1, video_patcher.num_patches, hidden_dim)
        )  # (1, N, d)
        self.pos_embed_prompt = nn.Parameter(
            torch.zeros(1, max_prompt_len, hidden_dim)
        )  # (1, L, d)
        
        
        # Projection of the diffusion time (so the model finds more useful projection than the sinusoidals)
        self.diffusion_timestep_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        self.layers = nn.ModuleList([
                TransformerBlock(
                    d_model=hidden_dim,
                    n_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, video_patcher.patch_dim)
        
        self._init_weights()


    def _init_weights(self):
        # Positional embeddings
        nn.init.normal_(self.pos_embed_video, std=0.02)
        nn.init.normal_(self.pos_embed_prompt, std=0.02)
        
        # Video projection
        nn.init.xavier_uniform_(self.video_proj.weight)
        nn.init.zeros_(self.video_proj.bias)
        
        # Output projection (zeros initialization to avoid bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)



    def forward(
            self,
            x_t: torch.Tensor,  # (B, T, C, H, W)
            diffusion_timestep: torch.Tensor,  # (B,)
            prompt_ids: torch.Tensor,  # (B, L)
        ) -> torch.Tensor:
        """
        Predicts the noise added epsilon_t to the videos x_t.
        """
        
        # Pre-process video
        x_t = self.video_patcher.patchify(x_t)
        x_t = self.video_proj(x_t)
        x_t = x_t + self.pos_embed_video
        
        # Pre-process prompt
        prompt_embed = self.token_embed(prompt_ids)
        prompt_embed = prompt_embed + self.pos_embed_prompt
        prompt_mask = prompt_ids == self.pad_token_id
        
        # Bias according to the diffusion timestep
        sinusoidal_proj = sinusoidal_diffusion_timestep_embedding(diffusion_timestep, self.hidden_dim)
        diffusion_timestep_bias = self.diffusion_timestep_proj(sinusoidal_proj).unsqueeze(1)  # to allow broadcasting
        x_t = x_t + diffusion_timestep_bias
        
        # OBS: It seems diffusion_timestep_bias should be added in every layer, not only the first one.

        # Apply transformer layers
        for layer in self.layers:
            x_t = layer(
                x=x_t,
                cond=prompt_embed,
                cond_mask=prompt_mask,
            )
        
        # Final norm
        x_t = self.final_norm(x_t)

        # Project back to the patch dimension
        x_t = self.output_proj(x_t)

        # Unpatchify
        eps_hat = self.video_patcher.unpatchify(x_t)
        return eps_hat


if __name__ == "__main__":
    print(torch.cuda.is_available())
    
    B = 1
    T, H, W = 16, 32, 32
    pt, ph, pw = 2, 4, 4
    C = 1
    
    # Config
    vocab_size = 36
    max_prompt_len = 10
    hidden_dim = 128
    mlp_ratio = 4
    num_heads = 8
    num_layers = 12
    pad_token_id = 0
    
    # Data
    device = torch.device("cuda:0")
    x_t = torch.randn(B, T, C, H, W, device=device)
    diffusion_timestep = torch.randint(0, 1000, (B,), device=device)
    prompt_ids = torch.randint(1, vocab_size, (B, max_prompt_len), device=device)
    
    prompt_ids[1, -5:] = 0  # Simulate padding
    

    # Model
    video_patcher = VideoPatcher(T=T, H=H, W=W, pt=pt, ph=ph, pw=pw, C=C).to(device)
    diffusion_transformer = DiffusionTransformer(
        video_patcher=video_patcher,
        prompt_vocab_size=vocab_size,
        max_prompt_len=max_prompt_len,
        hidden_dim=hidden_dim,
        mlp_ratio=mlp_ratio,
        num_heads=num_heads,
        num_layers=num_layers,
        pad_token_id=pad_token_id,
    ).to(device)
    
    eps_hat = diffusion_transformer(
        x_t=x_t,
        diffusion_timestep=diffusion_timestep,
        prompt_ids=prompt_ids,
    )
    
    print(f"x_t: {x_t.shape}")
    print(eps_hat.shape)
    
    print("Done")