from functools import cached_property

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
    def __init__(
            self,
            
            # Original
            T: int,  # time dimension
            H: int,  # height
            W: int,  # width
            
            # Patches
            pt: int = 2,  # time block size
            ph: int = 4,  # height block size
            pw: int = 4,  # width block size
            
            C: int = 1,  # channels
    ):
        super().__init__()
        assert T % pt == 0 and H % ph == 0 and W % pw == 0

        self.T, self.H, self.W = T, H, W
        self.pt, self.ph, self.pw = pt, ph, pw
        self.C = C

        # Number of blocks
        self.t_blocks = T // pt
        self.h_blocks = H // ph
        self.w_blocks = W // pw

    @cached_property
    def num_patches(self) -> int:
        return self.t_blocks * self.h_blocks * self.w_blocks

    @cached_property
    def patch_dim(self) -> int:
        return self.pt * self.ph * self.pw * self.C

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



# ---- example usage ----
if __name__ == "__main__":
    B, T, C, H, W = 1, 16, 1, 32, 32
    pt, ph, pw = 2, 16, 16
    
    # Expected number of patches
    num_patches = (T // pt) * (H // ph) * (W // pw)
    patch_dim = pt * ph * pw * C # Flattened block dim
    
    print(f"Initial block shape (T, H, W): {T, H, W}")
    print(f"Patches distro. t: {T // pt}, h: {H // ph}, w: {W // pw}")
    print(f"Expected patch dimension: {patch_dim}")
    
    x = torch.randn(B, T, C, H, W)
    patcher = VideoPatcher(T=T, H=H, W=W, pt=pt, ph=ph, pw=pw, C=C)

    # PATCHIFY
    patches = patcher.patchify(x)
    print(f"Patched block shape (B, N, patch_dim): {patches.shape}")
    assert patches.shape == (B, num_patches, patch_dim)
    
    # UNPATCHIFY
    unpatched = patcher.unpatchify(patches)
    print(f"Unpatched block shape (B, T, C, H, W): {unpatched.shape}")
    assert unpatched.shape == (B, T, C, H, W)
    assert torch.allclose(x, unpatched)
    print("Test passed")