from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_transforms(
    image_height: int = 900,
    image_width: int = 1500,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    train: bool = True,
) -> transforms.Compose:
    """Create image transforms.

    The notebook resized images to 900 x 1500 and normalized them using mean/std of 0.5.
    """
    transform_list = [
        transforms.Resize((image_height, image_width)),
    ]

    # Optional augmentation can be added here.
    # if train:
    #     transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return transforms.Compose(transform_list)


def build_imagefolder_dataset(
    root: Path,
    image_height: int = 900,
    image_width: int = 1500,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    train: bool = True,
) -> torchvision.datasets.ImageFolder:
    """Build an ImageFolder dataset."""
    transform = build_transforms(
        image_height=image_height,
        image_width=image_width,
        mean=mean,
        std=std,
        train=train,
    )
    return torchvision.datasets.ImageFolder(root=str(root), transform=transform)


def build_dataloaders(
    data_root: Path,
    train_folder: str,
    valid_folder: str,
    batch_size: int = 8,
    num_workers: int = 2,
    image_height: int = 900,
    image_width: int = 1500,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
):
    """Create training and validation dataloaders."""
    train_dataset = build_imagefolder_dataset(
        data_root / train_folder,
        image_height=image_height,
        image_width=image_width,
        mean=mean,
        std=std,
        train=True,
    )

    valid_dataset = build_imagefolder_dataset(
        data_root / valid_folder,
        image_height=image_height,
        image_width=image_width,
        mean=mean,
        std=std,
        train=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, valid_loader


def build_test_loader(
    data_root: Path,
    test_folder: str,
    batch_size: int = 8,
    num_workers: int = 2,
    image_height: int = 900,
    image_width: int = 1500,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
):
    """Create a test dataloader."""
    test_dataset = build_imagefolder_dataset(
        data_root / test_folder,
        image_height=image_height,
        image_width=image_width,
        mean=mean,
        std=std,
        train=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return test_loader
