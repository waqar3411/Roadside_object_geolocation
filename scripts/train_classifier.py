#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from mappillary_classifier.datasets import build_dataloaders
from mappillary_classifier.models import create_model
from mappillary_classifier.train import train_model
from mappillary_classifier.utils import get_device, save_json, set_seed
from mappillary_classifier.visualization import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet/SEResNet classifier.")

    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--train-folder", type=str, default="Train_1000")
    parser.add_argument("--valid-folder", type=str, default="Test_new")
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--model-name", type=str, default="seresnet34",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "seresnet34"])
    parser.add_argument("--run-name", type=str, default="resnet_classifier")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.1)

    parser.add_argument("--image-height", type=int, default=900)
    parser.add_argument("--image-width", type=int, default=1500)
    parser.add_argument("--mean", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    parser.add_argument("--std", type=float, nargs=3, default=(0.5, 0.5, 0.5))

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="AttentionResnet")

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader = build_dataloaders(
        data_root=Path(args.data_root),
        train_folder=args.train_folder,
        valid_folder=args.valid_folder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_height=args.image_height,
        image_width=args.image_width,
        mean=tuple(args.mean),
        std=tuple(args.std),
    )

    model = create_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma,
    )

    wandb_run = None

    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )

    history = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        output_dir=output_dir,
        run_name=args.run_name,
        wandb_run=wandb_run,
    )

    save_json(history, output_dir / f"{args.run_name}_history.json")

    plot_training_curves(
        train_loss=history["train_loss"],
        train_acc=history["train_acc"],
        valid_loss=history["valid_loss"],
        valid_acc=history["valid_acc"],
        title=args.run_name,
        output_dir=output_dir,
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
