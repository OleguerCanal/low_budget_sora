from __future__ import annotations

import glob
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from low_budget_sora import COMIC_SANS_MS_FONT_PATH

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


def make_sliding_chars(
    sequence: str,
    T: int = 16,
    H: int = 32,
    W: int = 32,
    randomize_font: bool = False,
    randomize_color: bool = False,
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

    text = "".join(str(char) for char in sequence)
    font = random_font(font_paths=font_paths) if randomize_font else\
        ImageFont.truetype(COMIC_SANS_MS_FONT_PATH, size=34)
    
    tmp = Image.new("L", (1, 1), 0)
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    margin = 2
    x_start = W + margin
    x_end = -tw - margin
    y = (H - th) // 2
    y = -10

    frames: list[np.ndarray] = []
    for t in range(T):
        x = int(x_start + (x_end - x_start) * t / (T - 1))
        img = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(img)
        color = random.randint(200, 255) if randomize_color else 255
        draw.text((x, y), text, font=font, fill=color)
        frames.append(np.array(img, dtype=np.uint8))

    return np.stack(frames, axis=0)


def save_gif(
    out_path: str | os.PathLike[str],
    clip: np.ndarray | Iterable[np.ndarray],
    *,
    fps: float = 5,
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
    imageio.mimsave(path.as_posix(), frames, format="GIF", fps=5)

def tensor_video_to_gif(video: torch.Tensor, path: str, fps: float = 5):
    """
    video: (T, 1, H, W) or (T, H, W), values in [0,1] or [0,255]
    path: output gif path, e.g. "debug.gif"
    """
    if video.dim() == 4:  # (T,1,H,W)
        video = video[:, 0, ...]  # -> (T,H,W)

    video = video.detach().cpu().float()
    vmin, vmax = video.min(), video.max()
    video = (video - vmin / (vmax - vmin)) * 255.0

    video = video.clamp(0, 255).byte().numpy()  # (T,H,W), uint8

    frames = [frame for frame in video]  # grayscale frames
    imageio.mimsave(path, frames, format="GIF", fps=fps)
    print(f"Saved GIF to {path}")
    
def console_print(frame):    
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if frame[i, j] > 128:
                print("⬜", end="")
            else:
                print("🟥", end="")
        print()

if __name__ == "__main__":
    # sequence = ["O", "L", "E", "G", "U", "E", "R"]
    sequence = "O"
    vid = make_sliding_chars(sequence)  # (16, 32, 32)
    console_print(vid[0])
    print()
    console_print(vid[1])
    print()
    console_print(vid[2])
    print()
    console_print(vid[3])
    save_gif("digits.gif", vid, fps=5)
    
    print("Percentage of white pixels: ", np.sum(vid > 128) / (16 * 32 * 32))
    print("Saved to digits.gif")