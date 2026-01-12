#!/usr/bin/env python3
"""
Visualize cosine similarity matrices for learned positional embeddings:
- pos_embed_video:  (1, N, d)
- pos_embed_prompt: (1, L, d)

Usage:
  python viz_pos_cosine.py --ckpt /path/to/checkpoint.pt
  python viz_pos_cosine.py --ckpt /path/to/lightning.ckpt --save_dir ./out

Notes:
- Expects keys: "pos_embed_video" and "pos_embed_prompt" in the loaded state_dict
  (or inside checkpoint["state_dict"]).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def load_state_dict(ckpt_path: str, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location=map_location)

    # Lightning usually stores weights under "state_dict"
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")

    # Some Lightning checkpoints prefix keys with "model." or similar.
    # We'll try direct first; fallback to stripping common prefixes.
    if "pos_embed_video" in sd and "pos_embed_prompt" in sd:
        return sd

    # Try stripping one prefix level
    def strip_prefix_once(key: str) -> str:
        return key.split(".", 1)[1] if "." in key else key

    stripped = {strip_prefix_once(k): v for k, v in sd.items()}
    if "pos_embed_video" in stripped and "pos_embed_prompt" in stripped:
        return stripped

    # Try stripping "model." specifically
    stripped_model = {k.replace("model.", "", 1): v for k, v in sd.items()}
    if "pos_embed_video" in stripped_model and "pos_embed_prompt" in stripped_model:
        return stripped_model

    # If still missing, show a helpful error.
    keys_preview = list(sd.keys())[:30]
    raise KeyError(
        "Could not find required keys 'pos_embed_video' and 'pos_embed_prompt' in checkpoint. "
        f"First keys: {keys_preview}"
    )


@torch.no_grad()
def cosine_sim_matrix(pos: torch.Tensor) -> torch.Tensor:
    """
    pos: (1, N, d) or (N, d)
    returns: (N, N) cosine similarity
    """
    if pos.dim() == 3:
        pos = pos[0]  # (N, d)
    pos = F.normalize(pos, dim=-1)
    return pos @ pos.t()  # (N, N)


def offdiag_stats(M: torch.Tensor) -> Tuple[float, float, float]:
    """
    Return (min, mean, max) over off-diagonal entries.
    """
    n = M.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=M.device)
    vals = M[mask]
    return float(vals.min().cpu()), float(vals.mean().cpu()), float(vals.max().cpu())


def plot_heatmap(M: torch.Tensor, title: str, ax: plt.Axes, vmax: float = 1.0, vmin: float = -1.0):
    im = ax.imshow(M.cpu().numpy(), aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("position index")
    ax.set_ylabel("position index")
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_location", type=str, default="cpu", help="cpu | cuda:0 | etc.")
    parser.add_argument("--save_dir", type=str, default=".", help="If set, saves PNGs there.")
    parser.add_argument("--center", action="store_true", help="Use diverging colormap range centered on 0.")
    args = parser.parse_args()

    ckpt = "/home/mary/code/oleguer/low_budget_sora/src/low_budget_sora/checkpoints/last-v2.ckpt"
    sd = load_state_dict(ckpt, map_location=args.map_location)

    pos_v = sd["pos_embed_video"]   # (1, N, d)
    pos_p = sd["pos_embed_prompt"]  # (1, L, d)

    Mv = cosine_sim_matrix(pos_v)
    Mp = cosine_sim_matrix(pos_p)

    vmin, vmax = (-1.0, 1.0) if args.center else (float(min(Mv.min(), Mp.min())), float(max(Mv.max(), Mp.max())))

    mv_stats = offdiag_stats(Mv)
    mp_stats = offdiag_stats(Mp)

    print(f"[video]  shape pos={tuple(pos_v.shape)}  sim={tuple(Mv.shape)}  offdiag min/mean/max={mv_stats}")
    print(f"[prompt] shape pos={tuple(pos_p.shape)}  sim={tuple(Mp.shape)}  offdiag min/mean/max={mp_stats}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    im0 = plot_heatmap(Mv, f"Video pos_embed cosine sim (N={Mv.shape[0]})", axes[0], vmin=vmin, vmax=vmax)
    im1 = plot_heatmap(Mp, f"Prompt pos_embed cosine sim (L={Mp.shape[0]})", axes[1], vmin=vmin, vmax=vmax)

    # One shared colorbar
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label("cosine similarity")

    # Optional save
    if args.save_dir:
        outdir = Path(args.save_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        figpath = outdir / "positional_cosine_similarity.png"
        fig.savefig(figpath, dpi=200)
        print(f"Saved: {figpath}")

        # Also save separate matrices as images if desired
        # (useful for quick inspection in logs)
        plt.figure(figsize=(7, 6))
        plt.imshow(Mv.cpu().numpy(), aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
        plt.title("Video positional cosine sim")
        plt.xlabel("position index"); plt.ylabel("position index")
        plt.colorbar(label="cosine similarity")
        vpath = outdir / "video_pos_cosine.png"
        plt.savefig(vpath, dpi=200)
        plt.close()
        print(f"Saved: {vpath}")

        plt.figure(figsize=(7, 6))
        plt.imshow(Mp.cpu().numpy(), aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
        plt.title("Prompt positional cosine sim")
        plt.xlabel("position index"); plt.ylabel("position index")
        plt.colorbar(label="cosine similarity")
        ppath = outdir / "prompt_pos_cosine.png"
        plt.savefig(ppath, dpi=200)
        plt.close()
        print(f"Saved: {ppath}")


if __name__ == "__main__":
    main()
