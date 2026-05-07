#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from mappillary_classifier.iou_evaluation import evaluate_gradcam_iou
from mappillary_classifier.models import create_model
from mappillary_classifier.utils import get_device, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Grad-CAM masks against ground-truth masks.")

    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--mask-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="resnet34",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "seresnet34"])
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--target-class", type=int, default=1)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--no-pretrained", action="store_true")

    parser.add_argument("--mean", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    parser.add_argument("--std", type=float, nargs=3, default=(0.5, 0.5, 0.5))

    parser.add_argument("--device", type=str, default=None)

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

    summary = evaluate_gradcam_iou(
        model=model,
        model_name=args.model_name,
        image_dir=Path(args.image_dir),
        mask_dir=Path(args.mask_dir),
        output_dir=Path(args.output_dir),
        device=device,
        target_class=args.target_class,
        iou_threshold=args.iou_threshold,
        mean=tuple(args.mean),
        std=tuple(args.std),
    )

    print(summary)


if __name__ == "__main__":
    main()
