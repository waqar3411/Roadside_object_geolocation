from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from tqdm import tqdm

from .gradcam_localization import preprocess_for_gradcam, threshold_cam_otsu
from .models import get_gradcam_target_layer
from .utils import ensure_dir, save_json

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    GradCAM = None
    ClassifierOutputTarget = None


def iter_image_files(image_dir: Path, extensions=(".jpg", ".jpeg", ".png")) -> Iterable[Path]:
    """Yield image files."""
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() in extensions:
            yield path


def read_binary_mask(mask_path: Path) -> np.ndarray:
    """Read mask as binary array."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    return (mask > 0).astype(np.uint8)


def connected_components(binary_mask: np.ndarray) -> List[np.ndarray]:
    """Return connected component masks."""
    labels, num = ndi.label(binary_mask)

    components: List[np.ndarray] = []

    for label_id in range(1, num + 1):
        component = labels == label_id
        if component.sum() > 0:
            components.append(component)

    return components


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def match_components(
    gt_components: List[np.ndarray],
    pred_components: List[np.ndarray],
    iou_threshold: float,
) -> Tuple[int, int, int, List[float]]:
    """Greedy component matching using IoU."""
    matched_pred = set()
    ious: List[float] = []
    tp = 0

    for gt in gt_components:
        best_iou = 0.0
        best_idx = None

        for pred_idx, pred in enumerate(pred_components):
            if pred_idx in matched_pred:
                continue

            iou = compute_iou(gt, pred)

            if iou > best_iou:
                best_iou = iou
                best_idx = pred_idx

        if best_idx is not None and best_iou >= iou_threshold:
            tp += 1
            matched_pred.add(best_idx)
            ious.append(best_iou)

    fp = len(pred_components) - len(matched_pred)
    fn = len(gt_components) - tp

    return tp, fp, fn, ious


def generate_cam_mask(
    model: torch.nn.Module,
    model_name: str,
    image_path: Path,
    device: torch.device,
    target_class: int,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
) -> np.ndarray:
    """Generate binary Grad-CAM mask for one image."""
    if GradCAM is None:
        raise ImportError("Please install grad-cam: pip install grad-cam")

    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess_for_gradcam(image, mean=mean, std=std).to(device)

    target_layer = get_gradcam_target_layer(model, model_name)
    targets = [ClassifierOutputTarget(target_class)]

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    return threshold_cam_otsu(grayscale_cam)


def evaluate_gradcam_iou(
    model: torch.nn.Module,
    model_name: str,
    image_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    device: torch.device,
    target_class: int = 1,
    iou_threshold: float = 0.5,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
) -> Dict[str, float]:
    """Evaluate Grad-CAM masks against ground-truth masks."""
    ensure_dir(output_dir)

    model = model.to(device)
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious: List[float] = []
    per_image_rows = []

    for image_path in tqdm(list(iter_image_files(image_dir)), desc="IoU evaluation"):
        mask_path = mask_dir / image_path.name

        if not mask_path.exists():
            continue

        gt_mask = read_binary_mask(mask_path)
        pred_mask = generate_cam_mask(
            model=model,
            model_name=model_name,
            image_path=image_path,
            device=device,
            target_class=target_class,
            mean=mean,
            std=std,
        )

        gt_components = connected_components(gt_mask)
        pred_components = connected_components(pred_mask)

        tp, fp, fn, ious = match_components(
            gt_components=gt_components,
            pred_components=pred_components,
            iou_threshold=iou_threshold,
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(ious)

        per_image_rows.append(
            {
                "filename": image_path.name,
                "gt_components": len(gt_components),
                "pred_components": len(pred_components),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "mean_matched_iou": float(np.mean(ious)) if ious else 0.0,
            }
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    summary = {
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "precision": float(precision),
        "recall": float(recall),
        "detection_rate": float(recall),
        "mean_matched_iou": float(mean_iou),
        "iou_threshold": float(iou_threshold),
    }

    save_json(summary, output_dir / "iou_summary.json")

    with open(output_dir / "iou_per_image.csv", "w", newline="") as csvfile:
        fieldnames = [
            "filename",
            "gt_components",
            "pred_components",
            "tp",
            "fp",
            "fn",
            "mean_matched_iou",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)

    return summary
