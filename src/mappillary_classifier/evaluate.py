from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torchvision

from .utils import ensure_dir, save_json


def unnormalize_tensor_image(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> torch.Tensor:
    """Undo normalization for image saving."""
    image = tensor.detach().cpu().clone()

    for channel in range(3):
        image[channel] = image[channel] * std[channel] + mean[channel]

    return image.clamp(0, 1)


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    class1_probs: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Calculate classification metrics."""
    metrics = {
        "accuracy": float(sklearn.metrics.accuracy_score(y_true, y_pred)),
        "precision": float(sklearn.metrics.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(sklearn.metrics.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(sklearn.metrics.f1_score(y_true, y_pred, zero_division=0)),
    }

    if class1_probs is not None and len(set(y_true)) > 1:
        metrics["auc"] = float(sklearn.metrics.roc_auc_score(y_true, class1_probs))
    else:
        metrics["auc"] = float("nan")

    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics.update(
        {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }
    )

    return metrics


def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    output_dir: Optional[Path] = None,
    save_prediction_images: bool = False,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> Dict[str, float]:
    """Evaluate classifier and optionally save predicted images."""
    model.eval()

    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    class1_probs: List[float] = []

    if output_dir is not None:
        ensure_dir(output_dir)

    if save_prediction_images and output_dir is not None:
        ensure_dir(output_dir / "predicted_0")
        ensure_dir(output_dir / "predicted_1")

    image_counter = 0

    with torch.no_grad():
        for batch, target in loader:
            batch = batch.to(device)
            target = target.to(device)

            logits = model(batch)
            loss = criterion(logits, target)
            probabilities = torch.softmax(logits, dim=1)

            predictions = torch.argmax(probabilities, dim=1)

            losses.append(float(loss.item()))
            y_true.extend(target.detach().cpu().numpy().tolist())
            y_pred.extend(predictions.detach().cpu().numpy().tolist())
            class1_probs.extend(probabilities[:, 1].detach().cpu().numpy().tolist())

            if save_prediction_images and output_dir is not None:
                for idx in range(batch.size(0)):
                    image_counter += 1
                    true_label = int(target[idx].item())
                    pred_label = int(predictions[idx].item())

                    image = unnormalize_tensor_image(batch[idx], mean=mean, std=std)
                    pil_image = torchvision.transforms.functional.to_pil_image(image)

                    save_name = f"{image_counter}_true_{true_label}_pred_{pred_label}.jpg"
                    save_path = output_dir / f"predicted_{pred_label}" / save_name
                    pil_image.save(save_path)

    metrics = calculate_metrics(y_true, y_pred, class1_probs)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0

    if output_dir is not None:
        save_json(metrics, output_dir / "metrics.json")

        cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
        with open(output_dir / "confusion_matrix.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["", "pred_0", "pred_1"])
            writer.writerow(["true_0", int(cm[0, 0]), int(cm[0, 1])])
            writer.writerow(["true_1", int(cm[1, 0]), int(cm[1, 1])])

    return metrics
