import torch
import torch.nn as nn

class VideoPatcher(nn.Module):
    """
    3D patcher for videos.

    Input:  (B, T, C, H, W)
    Output: (B, N, patch_dim) where
        N = (T/pt) * (H/ph) * (W/pw)
        patch_dim = pt * ph * pw * C
    """
    def __init__(self, T, H, W, pt=2, ph=4, pw=4, C=1):
        super().__init__()
        assert T % pt == 0 and H % ph == 0 and W % pw == 0

        self.T, self.H, self.W = T, H, W
        self.pt, self.ph, self.pw = pt, ph, pw
        self.C = C

        self.t_blocks = T // pt
        self.h_blocks = H // ph
        self.w_blocks = W // pw

        self.num_patches = self.t_blocks * self.h_blocks * self.w_blocks
        self.patch_dim = pt * ph * pw * C

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        returns: (B, N, patch_dim)
        """
        B, T, C, H, W = x.shape
        assert T == self.T and H == self.H and W == self.W and C == self.C

        x = x.view(
            B,
            self.t_blocks, self.pt,
            C,
            self.h_blocks, self.ph,
            self.w_blocks, self.pw,
        )  # (B, tB, pt, C, hB, ph, wB, pw)

        # bring blocks together then flatten
        x = x.permute(0, 1, 4, 6, 2, 5, 7, 3)  # (B, tB, hB, wB, pt, ph, pw, C)
        x = x.reshape(B, self.num_patches, self.patch_dim)
        return x

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, N, patch_dim)
        returns: (B, T, C, H, W)
        """
        B, N, D = tokens.shape
        assert N == self.num_patches and D == self.patch_dim

        x = tokens.view(
            B,
            self.t_blocks, self.h_blocks, self.w_blocks,
            self.pt, self.ph, self.pw, self.C,
        )  # (B, tB, hB, wB, pt, ph, pw, C)

        x = x.permute(0, 1, 4, 7, 2, 5, 3, 6)  # (B, tB, pt, C, hB, ph, wB, pw)
        x = x.reshape(B, self.T, self.C, self.H, self.W)
        return x
