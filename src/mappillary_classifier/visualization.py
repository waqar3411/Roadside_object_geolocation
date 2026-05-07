from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_curves(
    train_loss: List[float],
    train_acc: List[float],
    valid_loss: List[float],
    valid_acc: List[float],
    title: str,
    output_dir: Path,
) -> None:
    """Save training and validation curves."""
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, marker="o", label="Train Accuracy")
    plt.plot(epochs, valid_acc, marker="o", label="Validation Accuracy")
    plt.grid(True)
    plt.title(f"Accuracy Curves: {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curves.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    plt.plot(epochs, valid_loss, marker="o", label="Validation Loss")
    plt.grid(True)
    plt.title(f"Loss Curves: {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=300)
    plt.close()


def save_confusion_matrix_plot(
    y_true,
    y_pred,
    output_path: Path,
    labels=("0", "1"),
) -> None:
    """Save a confusion matrix heatmap."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(7, 5))
    sns.set(font_scale=1.2)
    sns.heatmap(
        matrix,
        cmap="coolwarm",
        linecolor="white",
        linewidths=1,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt="d",
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
