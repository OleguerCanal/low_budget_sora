from __future__ import annotations

import glob
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _default_font_paths() -> list[str]:
    """
    Best-effort font discovery.

    We keep it intentionally simple: if we can't find system fonts, PIL's default
    bitmap font will be used instead.
    """
    candidates = [
        "/usr/share/fonts/truetype",  # many Linux distros
        "/usr/share/fonts",  # fallback
    ]
    paths: list[str] = []
    for font_dir in candidates:
        if os.path.isdir(font_dir):
            paths.extend(glob.glob(os.path.join(font_dir, "**/*.ttf"), recursive=True))
    return paths


def random_font(
    *,
    size_min: int = 16,
    size_max: int = 36,
    font_paths: Sequence[str] | None = None,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = random.randint(size_min, size_max)
    paths = list(font_paths) if font_paths is not None else _default_font_paths()
    if paths:
        path = random.choice(paths)
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            # Corrupt or unsupported font; fall through to default.
            pass
    return ImageFont.load_default()


def make_sliding_digits(
    nums: Sequence[int] | Sequence[str],
    *,
    T: int = 16,
    H: int = 32,
    W: int = 32,
    seed: int | None = None,
    font_paths: Sequence[str] | None = None,
) -> np.ndarray:
    """
    Generate a grayscale clip with digits sliding horizontally across frame.

    Returns: np.ndarray of shape (T, H, W), dtype uint8.
    """
    if T < 2:
        raise ValueError("T must be >= 2")
    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive")

    if seed is not None:
        random.seed(seed)

    text = "".join(str(n) for n in nums)
    font = random_font(font_paths=font_paths)

    tmp = Image.new("L", (1, 1), 0)
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    margin = 2
    x_start = W + margin
    x_end = -tw - margin
    y = (H - th) // 2

    frames: list[np.ndarray] = []
    for t in range(T):
        x = int(x_start + (x_end - x_start) * t / (T - 1))
        img = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(img)
        color = random.randint(200, 255)
        draw.text((x, y), text, font=font, fill=color)
        frames.append(np.array(img, dtype=np.uint8))

    return np.stack(frames, axis=0)


def save_gif(
    out_path: str | os.PathLike[str],
    clip: np.ndarray | Iterable[np.ndarray],
    *,
    duration: float = 0.1,
) -> None:
    """
    Save a grayscale clip to a GIF.

    - clip can be (T, H, W) or an iterable of (H, W)/(H, W, C) frames.
    """
    path = Path(out_path)
    if isinstance(clip, np.ndarray):
        frames = [frame for frame in clip]
    else:
        frames = list(clip)
    imageio.mimsave(path.as_posix(), frames, duration=duration)\

def tensor_video_to_gif(video: torch.Tensor, path: str, duration: float = 0.1):
    """
    video: (T, 1, H, W) or (T, H, W), values in [0,1] or [0,255]
    path: output gif path, e.g. "debug.gif"
    """
    if video.dim() == 4:  # (T,1,H,W)
        video = video[:, 0]  # -> (T,H,W)

    video = video.detach().cpu().float()
    vmin, vmax = video.min(), video.max()
    if vmax <= 1.0:  # assume [0,1]
        video = video * 255.0

    video = video.clamp(0, 255).byte().numpy()  # (T,H,W), uint8

    frames = [frame for frame in video]  # grayscale frames
    imageio.mimsave(path, frames, duration=duration)
    print(f"Saved GIF to {path}")
    
if __name__ == "__main__":
    vid = make_sliding_digits([1, 2, 3, 4])  # (16, 32, 32)
    save_gif("digits.gif", vid, duration=0.1)
    print("Saved to digits.gif")