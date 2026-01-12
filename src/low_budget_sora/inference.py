import yaml
import torch
import os

from low_budget_sora import CONFIG_FILE_PATH, CHECKPOINTS_DIR
from low_budget_sora.data.sequence_sampler import SequenceSampler
from low_budget_sora.training.train import get_datasets, get_model  # the same get_model you used for training
from low_budget_sora.data.utils import tensor_video_to_gif  # wherever you put it

if __name__ == "__main__":

    ckpt_path = os.path.join(CHECKPOINTS_DIR, "last-v2.ckpt")  # or best ckpt

    # 1) load config and build fresh model
    with open(CONFIG_FILE_PATH, "r") as f:
        config = yaml.safe_load(f)

    pl_model = get_model(config)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pl_model.load_state_dict(ckpt["state_dict"])
    pl_model.eval()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pl_model.to(device)

    # 2) build prompt_ids for a specific sequence [1,2,3]
    _, val_dataset = get_datasets(config)
    
    example = val_dataset[0]
    # example = val_dataset.generate_custom("MELISSA")
    
    # 3) sample video
    with torch.no_grad():
        videos = pl_model.sample_videos(
            prompt_ids=example["prompt_ids"].unsqueeze(0),
            num_steps=None,
            eta=0.0,
        )  # (1, T, C, H, W)

    video = videos[0]  # (T,C,H,W)

    # 4) write GIF
    print("Sequence: ", example["sequence"])
    print("Prompt IDs: ", example["prompt_ids"])
    tensor_video_to_gif(video, "debug_generated.gif")
    tensor_video_to_gif(example["video"], "debug_expected.gif")
    print("Done")