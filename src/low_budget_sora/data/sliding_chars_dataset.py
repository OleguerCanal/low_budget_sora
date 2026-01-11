from typing import Literal

from torch.utils.data import Dataset
import torch
import random

from low_budget_sora.data.utils import make_sliding_chars, tensor_video_to_gif
from low_budget_sora.data.sequence_sampler import SequenceSampler

class SlidingCharsDataset(Dataset):
    def __init__(
        self,
        sampler: SequenceSampler,
        split: Literal["train", "val"] = "train",
        T=16,
        H=32,
        W=32,
        pad_token=0,
        prompt_len=8,
    ):
        self.sampler = sampler
        self.split = split
        self.T, self.H, self.W = T, H, W
        self.pad_token = pad_token
        self.prompt_len = prompt_len

    def __len__(self):
        if self.split == "train":
            return len(self.sampler)
        elif self.split == "val":
            return len(self.sampler.val_list)
        raise ValueError(f"Invalid split: {self.split}")

    def __getitem__(self, idx):
        sample = self.sampler[idx, self.split]              # e.g. [3, 1, 4]
        video = make_sliding_chars(
            sequence=sample["sequence"],
            T=self.T,
            H=self.H,
            W=self.W,
        )

        # normalize and add channel dim: (T, 1, H, W), float32 in [0,1]
        video = torch.from_numpy(video).float() / 255.0
        video = video.unsqueeze(1)

        # pad sequence to max_len with pad_token
        L = sample["length"]
        prompt = torch.full((self.prompt_len,), self.pad_token, dtype=torch.long)
        prompt[:L] = torch.tensor(sample["tokens"], dtype=torch.long)

        return {
            "video": video,       # (T, 1, H, W)
            "sequence": sample["sequence"],
            "prompt_ids": prompt, # (max_len,)
            "length": L,
        }


# ---- example usage ----
if __name__ == "__main__":
    sampler = SequenceSampler(
        num_training_examples=1_000_000,
        num_validation_examples=1_000,
        seed=42,
    )
    train_ds = SlidingCharsDataset(
        sampler=sampler,
        split="train",
    )
    print(f"Train dataset size: {len(train_ds)}")
    
    val_ds = SlidingCharsDataset(
        sampler=sampler,
        split="val",
    )
    print(f"Validation dataset size: {len(val_ds)}")
    
    sample = train_ds[0]
    print(sample["video"].shape, sample["prompt_ids"], sample["sequence"], sample["length"])
    tensor_video_to_gif(sample["video"], "debug_train.gif")
    print("Saved GIF to debug_train.gif")
    
    sample = val_ds[0]
    print(sample["video"].shape, sample["prompt_ids"], sample["sequence"], sample["length"])
    tensor_video_to_gif(sample["video"], "debug_val.gif")
    print("Saved GIF to debug_val.gif")