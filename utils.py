import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import albumentations as albu


class SegmentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_folder: Path = "binary_dataset",
        masks_folder: Path = "images_semantic",
        images_folder: Path = "original_images",
        split: str = "train",
        dataset_size: int = 300,
        transform: albu.Compose = None,
    ):
        self._data_folder = data_folder
        self._masks_folder = os.path.join(data_folder, masks_folder)
        self._images_folder = os.path.join(data_folder, images_folder)

        if split == "train":
            self._masks_files = sorted(os.listdir(self._masks_folder))[:dataset_size]
            self._images_files = sorted(os.listdir(self._images_folder))[:dataset_size]

        elif split == "val":
            train_size = len(os.listdir(self._images_folder)) - (2 * dataset_size)
            val_size = train_size + dataset_size
            self._masks_files = sorted(os.listdir(self._masks_folder))[
                train_size:val_size
            ]
            self._images_files = sorted(os.listdir(self._images_folder))[
                train_size:val_size
            ]

        elif split == "test":
            test_size = len(os.listdir(self._images_folder)) - dataset_size
            self._masks_files = sorted(os.listdir(self._masks_folder))[test_size:]
            self._images_files = sorted(os.listdir(self._images_folder))[test_size:]

        self._transform = transform

    def __getitem__(self, idx):
        image_filename = self._images_files[idx]
        mask_filename = self._masks_files[idx]

        image = Image.open(os.path.join(self._images_folder, image_filename))
        mask = Image.open(os.path.join(self._masks_folder, mask_filename))

        image = np.array(image)
        mask = np.array(mask)

        batch = {"image": image, "mask": mask}

        if self._transform:
            batch = self._transform(image=image, mask=mask)

        batch = albu.Normalize()(image=batch["image"], mask=batch["mask"])

        return batch["image"].transpose(2, 0, 1), batch["mask"]

    def __len__(self):
        return len(self._images_files)
