import os
from dotenv import load_dotenv
import albumentations as albu
import pytorch_lightning as pl
import wandb
from model import SegmentModule


TRAIN_TRANSFORMS = albu.Compose(
    [
        albu.Resize(736, 736),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.ElasticTransform(
            p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
        ),
        albu.RandomBrightnessContrast(p=0.3),
    ]
)

TEST_TRANSFORMS = albu.Compose([albu.Resize(736, 736)])

config = {
    "model_type": "Unet",
    "model_params": {
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
        "in_channels": 3,
        "classes": 1,
    },
    "criterion_type": "Dice",
    "optimizer_type": "Adam",
    "optimizer_params": {
        "lr": 3e-5,
    },
    "dataloader_params": {
        "batch_size": 2,
    },
    "dataset_train_params": {
        "data_folder": "data/raw",
        "masks_folder": "images_semantic",
        "images_folder": "original_images",
        "transform": TRAIN_TRANSFORMS,
        "dataset_size": 300,
    },
    "dataset_val_params": {
        "data_folder": "data/raw",
        "masks_folder": "images_semantic",
        "images_folder": "original_images",
        "transform": TEST_TRANSFORMS,
        "dataset_size": 50,
    },
    "dataset_test_params": {
        "data_folder": "data/raw",
        "masks_folder": "images_semantic",
        "images_folder": "original_images",
        "transform": TEST_TRANSFORMS,
        "dataset_size": 50,
    },
}

if __name__ == "__main__":
    load_dotenv(".env")
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    module = SegmentModule(config)
    logger = pl.loggers.WandbLogger(project="segment_test", name="test_script")
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger,
        log_every_n_steps=10,
        max_epochs=1,
    )
    trainer.fit(module)
    trainer.test()
