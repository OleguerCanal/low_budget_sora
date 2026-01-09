from torch.utils.data import Dataset
import torch
import random

from low_budget_sora.utils import make_sliding_digits, tensor_video_to_gif

class SlidingDigitsDataset(Dataset):
    def __init__(
        self,
        num_samples=10_000,
        T=16,
        H=32,
        W=32,
        min_len=1,
        max_len=5,
        digits=list(range(10)),
        pad_token=-1,
    ):
        self.num_samples = num_samples
        self.T, self.H, self.W = T, H, W
        self.min_len, self.max_len = min_len, max_len
        self.digits = digits
        self.pad_token = pad_token

    def __len__(self):
        return self.num_samples

    def _sample_sequence(self):
        L = random.randint(self.min_len, self.max_len)
        return [random.choice(self.digits) for _ in range(L)]

    def __getitem__(self, idx):
        seq = self._sample_sequence()              # e.g. [3, 1, 4]
        video = make_sliding_digits(seq, self.T, self.H, self.W)  # (T, H, W), uint8

        # normalize and add channel dim: (T, 1, H, W), float32 in [0,1]
        video = torch.from_numpy(video).float() / 255.0
        video = video.unsqueeze(1)

        # pad sequence to max_len with pad_token
        L = len(seq)
        prompt = torch.full((self.max_len,), self.pad_token, dtype=torch.long)
        prompt[:L] = torch.tensor(seq, dtype=torch.long)

        return {
            "video": video,       # (T, 1, H, W)
            "prompt_ids": prompt, # (max_len,)
            "length": L,
        }


# ---- example usage ----
if __name__ == "__main__":
    ds = SlidingDigitsDataset(num_samples=1000)
    sample = ds[0]
    print(sample["video"].shape, sample["prompt_ids"], sample["length"])
    tensor_video_to_gif(sample["video"], "debug.gif")
    print("Saved GIF to debug.gif")