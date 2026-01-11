from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
import lightning as L  # Lightning 2.x


def sliding_chars_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # stack fixed-shape tensors
    video = torch.stack([b["video"] for b in batch], dim=0)            # (B, T, 1, H, W)
    prompt_ids = torch.stack([b["prompt_ids"] for b in batch], dim=0)  # (B, prompt_len)
    length = torch.tensor([b["length"] for b in batch], dtype=torch.long)  # (B,)

    # keep non-tensors as lists
    sequence = [b["sequence"] for b in batch]

    # optional: mask for attention over prompt tokens
    prompt_len = prompt_ids.size(1)
    prompt_mask = torch.arange(prompt_len).unsqueeze(0) < length.unsqueeze(1)  # (B, prompt_len) bool

    return {
        "video": video,
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "length": length,
        "sequence": sequence,
    }


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        *,
        batch_size: int = 32,
        val_batch_size: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        shuffle_train: bool = True,
        drop_last: bool = True,
        prefetch_factor: Optional[int] = 4,
        collate_fn=sliding_chars_collate_fn,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = (
            persistent_workers if persistent_workers is not None else (num_workers > 0)
        )
        self.shuffle_train = shuffle_train
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self.collate_fn = collate_fn

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=self.collate_fn,
        )


if __name__ == "__main__":
    import yaml
    from low_budget_sora import CONFIG_FILE_PATH
    with open(CONFIG_FILE_PATH, 'r') as stream:
        config = yaml.safe_load(stream)
        
    from low_budget_sora.training.train import get_datasets
    train_ds, val_ds = get_datasets(config)
    
    dm = DataModule(
        train_ds, val_ds, 
        batch_size=config["training"]["batch_size"], 
        num_workers=config["training"]["num_workers"],
        prefetch_factor=config["training"]["prefetch_factor"],
    )
    batch = next(iter(dm.train_dataloader()))
    
    print("video shape: ", batch["video"].shape)
    print("prompt_ids shape: ", batch["prompt_ids"].shape)
    print("prompt_mask shape: ", batch["prompt_mask"].shape)
    print("length shape: ", batch["length"].shape)
    
    print(batch["prompt_ids"])
    print(batch["prompt_mask"])
    for s in batch["sequence"]:
        print(s)
    print(batch["length"])