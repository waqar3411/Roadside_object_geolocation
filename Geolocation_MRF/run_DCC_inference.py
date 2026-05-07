#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DCC image inference pipeline.

This script:
1. Loads DCC street-level images from nested folders.
2. Resizes and center-crops each image to a fixed input size.
3. Runs a trained PyTorch classifier.
4. Saves positive predictions into confidence-based folders.
5. Writes confidence scores to CSV files.

Author: Waqar Ahmad
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    input_dir: Path
    model_path: Path
    output_dir: Path

    positive_class: int = 1

    input_mean: Tuple[float, float, float] = (0.4380, 0.4564, 0.4541)
    input_std: Tuple[float, float, float] = (0.2478, 0.2557, 0.2774)

    resized_width: int = 1500
    resized_height: int = 1500
    final_crop_width: int = 1500
    final_crop_height: int = 900

    image_extension: str = ".jpg"

    # Original script only processes folders ending with "01".
    # Set to None if you want to process all folders.
    folder_suffix: Optional[str] = "01"


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_output_dirs(output_dir: Path) -> None:
    """Create confidence-based output folders."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for folder_name in ["5_to_6", "6_to_8", "8_to_9", "greater_9"]:
        (output_dir / folder_name).mkdir(parents=True, exist_ok=True)


def initialize_bucket_csvs(output_dir: Path) -> None:
    """Create bucket CSV files with headers."""
    for bucket in ["5_to_6", "6_to_8", "8_to_9", "greater_9"]:
        csv_path = output_dir / f"{bucket}.csv"

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "confidence"])


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """Load trained PyTorch model and move it to the selected device."""
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()
    return model


# -------------------------------------------------------------------------
# Image preprocessing
# -------------------------------------------------------------------------

def center_crop_image(
    image: Image.Image,
    crop_width: int,
    crop_height: int,
) -> Image.Image:
    """Center-crop a PIL image."""
    width, height = image.size

    left = (width - crop_width) // 2
    upper = (height - crop_height) // 2
    right = left + crop_width
    lower = upper + crop_height

    return image.crop((left, upper, right, lower))


def resize_and_crop_image(
    image: Image.Image,
    config: InferenceConfig,
) -> Image.Image:
    """
    Resize image to 1500 × 1500 and center-crop it to 1500 × 900.

    This follows the same preprocessing logic as the original DCC script.
    """
    resized_image = image.resize(
        (config.resized_width, config.resized_height)
    )

    cropped_image = center_crop_image(
        resized_image,
        config.final_crop_width,
        config.final_crop_height,
    )

    return cropped_image


def preprocess_image(
    image: Image.Image,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> torch.Tensor:
    """Convert PIL image to normalized PyTorch tensor."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return transform(image).unsqueeze(0)


# -------------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------------

def predict_image(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[int, float]:
    """
    Run model prediction.

    Returns:
        predicted_class: predicted class index
        confidence: softmax confidence of predicted class
    """
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)

    predicted_class = int(torch.argmax(probabilities, dim=1).item())
    confidence = float(torch.max(probabilities, dim=1).values.item())

    return predicted_class, confidence


def get_confidence_bucket(confidence: float) -> Optional[str]:
    """Return confidence folder name."""
    if 0.5 <= confidence <= 0.6:
        return "5_to_6"

    if 0.6 < confidence <= 0.8:
        return "6_to_8"

    if 0.8 < confidence <= 0.9:
        return "8_to_9"

    if confidence > 0.9:
        return "greater_9"

    return None


# -------------------------------------------------------------------------
# Saving outputs
# -------------------------------------------------------------------------

def save_positive_prediction(
    image: Image.Image,
    filename: str,
    confidence: float,
    bucket: str,
    output_dir: Path,
) -> None:
    """Save detected image and append its confidence score to CSV."""
    image_output_path = output_dir / bucket / filename
    csv_output_path = output_dir / f"{bucket}.csv"

    image.save(image_output_path)

    with open(csv_output_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([filename, confidence])


# -------------------------------------------------------------------------
# Dataset traversal
# -------------------------------------------------------------------------

def should_process_folder(folder_path: Path, config: InferenceConfig) -> bool:
    """
    Decide whether a folder should be processed.

    The original script only processed folders whose names end with '01'.
    """
    if not folder_path.is_dir():
        return False

    if config.folder_suffix is None:
        return True

    return folder_path.name.endswith(config.folder_suffix)


def iter_image_paths(config: InferenceConfig):
    """
    Yield image paths from selected folders.

    Expected original structure:
        input_dir/
        ├── folder_01/
        │   ├── image1.jpg
        │   ├── image2.jpg
        ├── folder_02/
        │   ├── image3.jpg
    """
    for folder_path in sorted(config.input_dir.iterdir()):
        if not should_process_folder(folder_path, config):
            continue

        for image_path in sorted(folder_path.iterdir()):
            if image_path.suffix.lower() == config.image_extension.lower():
                yield image_path


# -------------------------------------------------------------------------
# Main inference pipeline
# -------------------------------------------------------------------------

def process_single_image(
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    config: InferenceConfig,
) -> None:
    """Process one image through preprocessing, prediction, and saving."""
    image = Image.open(image_path).convert("RGB")

    processed_image = resize_and_crop_image(image, config)

    image_tensor = preprocess_image(
        processed_image,
        config.input_mean,
        config.input_std,
    )

    predicted_class, confidence = predict_image(
        model,
        image_tensor,
        device,
    )

    if predicted_class != config.positive_class:
        return

    bucket = get_confidence_bucket(confidence)

    if bucket is None:
        return

    save_positive_prediction(
        image=processed_image,
        filename=image_path.name,
        confidence=confidence,
        bucket=bucket,
        output_dir=config.output_dir,
    )


def run_inference(config: InferenceConfig) -> None:
    """Run inference on all selected DCC images."""
    device = get_device()
    print(f"Using device: {device}")

    ensure_output_dirs(config.output_dir)
    initialize_bucket_csvs(config.output_dir)

    model = load_model(config.model_path, device)

    image_paths = list(iter_image_paths(config))

    print(f"Found {len(image_paths)} images to process.")

    for image_path in tqdm(image_paths):
        process_single_image(
            image_path=image_path,
            model=model,
            device=device,
            config=config,
        )

    print(f"Inference completed. Results saved to: {config.output_dir}")


# -------------------------------------------------------------------------
# Command-line interface
# -------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DCC image inference using a trained PyTorch classifier."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Root directory containing DCC image folders.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained PyTorch model file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where predictions and CSV files will be saved.",
    )

    parser.add_argument(
        "--folder-suffix",
        type=str,
        default="01",
        help=(
            "Only process folders ending with this suffix. "
            "Default is '01' to match the original script. "
            "Use 'all' to process all folders."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    folder_suffix = None if args.folder_suffix.lower() == "all" else args.folder_suffix

    config = InferenceConfig(
        input_dir=Path(args.input_dir),
        model_path=Path(args.model_path),
        output_dir=Path(args.output_dir),
        folder_suffix=folder_suffix,
    )

    run_inference(config)


if __name__ == "__main__":
    main()
