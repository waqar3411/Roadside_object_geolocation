from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from tqdm import tqdm

from .models import get_gradcam_target_layer
from .utils import ensure_dir


def iter_image_files(image_dir: Path, extensions=(".jpg", ".jpeg", ".png")) -> Iterable[Path]:
    """Yield image files from a directory."""
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() in extensions:
            yield path


def preprocess_for_gradcam(
    image: Image.Image,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
) -> torch.Tensor:
    """Convert PIL image to normalized tensor for Grad-CAM."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transform(image).unsqueeze(0)


def threshold_cam_otsu(grayscale_cam: np.ndarray) -> np.ndarray:
    """Threshold Grad-CAM map using Otsu method."""
    cam_uint8 = (grayscale_cam * 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(
        cam_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return binary_mask


def watershed_from_mask(
    binary_mask: np.ndarray,
    min_distance: int = 300,
    erosion_kernel_size: int = 100,
) -> np.ndarray:
    """Apply distance-transform watershed to split CAM regions."""
    distance = ndi.distance_transform_edt(binary_mask)

    distance_uint8 = distance.astype(np.uint8)
    _, distance_binary = cv2.threshold(
        distance_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    if erosion_kernel_size > 0:
        kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
        distance_binary = cv2.erode(distance_binary, kernel)

    coords = peak_local_max(distance, min_distance=min_distance, labels=distance_binary)

    if len(coords) == 0:
        return np.zeros_like(binary_mask, dtype=np.int32)

    marker_mask = np.zeros(distance.shape, dtype=bool)
    marker_mask[tuple(coords.T)] = True

    markers, _ = ndi.label(marker_mask)
    labels = watershed(distance, markers, mask=distance_binary.astype(bool))

    return labels


def generate_gradcam_for_directory(
    model: torch.nn.Module,
    model_name: str,
    image_dir: Path,
    output_dir: Path,
    device: torch.device,
    target_class: int = 1,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    save_binary_mask: bool = True,
    save_overlay: bool = True,
    save_watershed: bool = True,
) -> None:
    """Generate Grad-CAM maps for all images in a directory."""
    output_dir = ensure_dir(output_dir)
    overlay_dir = ensure_dir(output_dir / "overlays")
    mask_dir = ensure_dir(output_dir / "binary_masks")
    watershed_dir = ensure_dir(output_dir / "watershed_masks")

    model = model.to(device)
    model.eval()

    target_layer = get_gradcam_target_layer(model, model_name)
    targets = [ClassifierOutputTarget(target_class)]

    cam = GradCAM(model=model, target_layers=[target_layer])

    for image_path in tqdm(list(iter_image_files(image_dir)), desc="Grad-CAM"):
        image = Image.open(image_path).convert("RGB")
        image_float = np.float32(image) / 255.0

        input_tensor = preprocess_for_gradcam(image, mean=mean, std=std).to(device)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        if save_overlay:
            visualization = show_cam_on_image(image_float, grayscale_cam, use_rgb=True)
            Image.fromarray(visualization).save(overlay_dir / image_path.name)

        if save_binary_mask:
            binary_mask = threshold_cam_otsu(grayscale_cam)
            cv2.imwrite(str(mask_dir / image_path.name), binary_mask)

        if save_watershed:
            binary_mask = threshold_cam_otsu(grayscale_cam)
            watershed_labels = watershed_from_mask(binary_mask)
            watershed_vis = (watershed_labels > 0).astype(np.uint8) * 255
            cv2.imwrite(str(watershed_dir / image_path.name), watershed_vis)
