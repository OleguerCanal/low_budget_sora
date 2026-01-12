import os
import yaml

import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from low_budget_sora import CONFIG_FILE_PATH, CHECKPOINTS_DIR
from low_budget_sora.data.sequence_sampler import SequenceSampler
from low_budget_sora.data.sliding_chars_dataset import SlidingCharsDataset
from low_budget_sora.data.datamodule import DataModule
from low_budget_sora.model.video_patcher import VideoPatcher
from low_budget_sora.model.diffusion_transformer import DiffusionTransformer
from low_budget_sora.model.pl_model import DiffusionTransformerLitModule
from low_budget_sora.model.noise_scheduler import NoiseScheduler



def get_datasets(config: dict):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    sampler = SequenceSampler(
        num_training_examples=config["data"]["n_train_samples"],
        num_validation_examples=config["data"]["n_val_samples"],
        val_seed=config["data"]["seed"],
        train_seed=local_rank,
    )
    train_ds = SlidingCharsDataset(
        sampler=sampler,
        split="train",
        T=config["shapes"]["video"]["T"],
        H=config["shapes"]["video"]["H"],
        W=config["shapes"]["video"]["W"],
        prompt_len=config["data"]["prompt_len"],
    )
    
    val_ds = SlidingCharsDataset(
        sampler=sampler,
        split="val",
        T=config["shapes"]["video"]["T"],
        H=config["shapes"]["video"]["H"],
        W=config["shapes"]["video"]["W"],
        prompt_len=config["data"]["prompt_len"],
    )
    print(f"📚 Train dataset size: {len(train_ds)}")
    print(f"📚 Val dataset size: {len(val_ds)}")
    return train_ds, val_ds


def get_model(config: dict):
    video_patcher_params = {**config["shapes"]["video"], **config["shapes"]["patches"]}
    video_patcher = VideoPatcher(**video_patcher_params)
    
    diffusion_transformer = DiffusionTransformer(
        video_patcher=video_patcher,
        prompt_vocab_size=config["model"]["prompt_vocab_size"],
        max_prompt_len=config["data"]["prompt_len"],
        hidden_dim=config["model"]["hidden_dim"],
        mlp_ratio=config["model"]["mlp_ratio"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        pad_token_id=config["data"]["pad_token"],
    )
    
    noise_schedule = NoiseScheduler(
        num_steps=config["model"]["noise_scheduler_steps"],
    )
    
    pl_model = DiffusionTransformerLitModule(
        model=diffusion_transformer,
        noise_schedule=noise_schedule,
        lr=config["training"]["lr"],
    )
    return pl_model

if __name__ == "__main__":    
    # Config
    with open(CONFIG_FILE_PATH, 'r') as stream:
        config = yaml.safe_load(stream)
    
    L.seed_everything(config["data"]["seed"], workers=True)
    
    # Data
    train_ds, val_ds = get_datasets(config)
    datamodule = DataModule(
        train_dataset=train_ds,
        val_dataset=val_ds, 
        batch_size=config["training"]["batch_size"], 
        num_workers=config["training"]["num_workers"],
        prefetch_factor=config["training"]["prefetch_factor"],
    )

    # Model
    pl_model = get_model(config)
    
    # Logger
    wandb_logger = L.pytorch.loggers.WandbLogger(
        entity="oleguer_canal-waveshot",
        project="low-budget-sora",
        name=None,  # auto-generates a run name, or set your own
        config=config,  # logs your full config to wandb
    )
    
    # Train
    trainer = Trainer(
        max_epochs=config["training"]["max_epochs"],
        devices=config["training"]["devices"],
        accelerator="gpu",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        precision=config["training"]["precision"],
        default_root_dir=CHECKPOINTS_DIR,
        log_every_n_steps=10,
        val_check_interval=0.1,
        logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=True,
                dirpath=CHECKPOINTS_DIR,
                filename="{epoch:02d}-{val_loss:.4f}",
            )
        ],
    )
    trainer.fit(pl_model, datamodule)
    
    print("Done training")