from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import ensure_dir, save_checkpoint


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    wandb_run=None,
) -> Tuple[float, float]:
    """Train model for one epoch."""
    model.train()

    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for batch, target in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        logits = model(batch)
        loss = criterion(logits, target)

        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

        predictions = torch.argmax(logits, dim=1)
        y_true.extend(target.detach().cpu().numpy().tolist())
        y_pred.extend(predictions.detach().cpu().numpy().tolist())

        if wandb_run is not None:
            wandb_run.log({"batch_loss": float(loss.item())})

    avg_loss = float(np.mean(losses)) if losses else 0.0
    accuracy = float(sklearn.metrics.accuracy_score(y_true, y_pred))

    return avg_loss, accuracy


def validate_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate model for one epoch."""
    model.eval()

    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for batch, target in tqdm(loader, desc="Validation", leave=False):
            batch = batch.to(device)
            target = target.to(device)

            logits = model(batch)
            loss = criterion(logits, target)

            losses.append(float(loss.item()))

            predictions = torch.argmax(logits, dim=1)
            y_true.extend(target.detach().cpu().numpy().tolist())
            y_pred.extend(predictions.detach().cpu().numpy().tolist())

    avg_loss = float(np.mean(losses)) if losses else 0.0
    accuracy = float(sklearn.metrics.accuracy_score(y_true, y_pred))

    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epochs: int,
    output_dir: Path,
    run_name: str,
    wandb_run=None,
) -> Dict[str, List[float]]:
    """Full training loop with best-loss and best-accuracy checkpoints."""
    ensure_dir(output_dir)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
    }

    best_valid_loss = float("inf")
    best_valid_acc = -float("inf")

    model = model.to(device)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            wandb_run=wandb_run,
        )

        valid_loss, valid_acc = validate_one_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
        }

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, **metrics})

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(
                model=model,
                path=output_dir / f"{run_name}_best_loss.pt",
                epoch=epoch,
                metrics=metrics,
            )

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            save_checkpoint(
                model=model,
                path=output_dir / f"{run_name}_best_acc.pt",
                epoch=epoch,
                metrics=metrics,
            )

    save_checkpoint(
        model=model,
        path=output_dir / f"{run_name}_last.pt",
        epoch=epochs,
        metrics={
            "train_loss": history["train_loss"][-1],
            "train_acc": history["train_acc"][-1],
            "valid_loss": history["valid_loss"][-1],
            "valid_acc": history["valid_acc"][-1],
        },
    )

    return history
