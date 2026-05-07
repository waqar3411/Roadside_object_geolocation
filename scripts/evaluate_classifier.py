#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch.nn as nn

from mappillary_classifier.datasets import build_test_loader
from mappillary_classifier.evaluate import evaluate_model
from mappillary_classifier.models import create_model
from mappillary_classifier.utils import get_device, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier.")

    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--test-folder", type=str, default="Test_new")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="resnet34",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "seresnet34"])
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")

    parser.add_argument("--image-height", type=int, default=900)
    parser.add_argument("--image-width", type=int, default=1500)
    parser.add_argument("--mean", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    parser.add_argument("--std", type=float, nargs=3, default=(0.5, 0.5, 0.5))

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-prediction-images", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device(args.device)

    model = create_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
    )

    model = load_checkpoint(
        model=model,
        checkpoint_path=Path(args.model_path),
        device=device,
    )

    test_loader = build_test_loader(
        data_root=Path(args.data_root),
        test_folder=args.test_folder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_height=args.image_height,
        image_width=args.image_width,
        mean=tuple(args.mean),
        std=tuple(args.std),
    )

    metrics = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        output_dir=Path(args.output_dir),
        save_prediction_images=args.save_prediction_images,
        mean=tuple(args.mean),
        std=tuple(args.std),
    )

    print(metrics)


if __name__ == "__main__":
    main()
