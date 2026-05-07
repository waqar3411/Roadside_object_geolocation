# Mapillary Roadside Object Classification and Grad-CAM Localization

This repository contains a modular PyTorch pipeline for weakly supervised roadside object localization from street-level imagery.

The notebook workflow has been separated into independent components for:

1. Training a ResNet/SEResNet binary classifier.
2. Evaluating classification performance.
3. Saving predicted test images by class.
4. Generating Grad-CAM localization maps.
5. Evaluating Grad-CAM masks against ground-truth segmentation masks using IoU-style matching.

## Repository structure

```text
mappillary_gradcam_classifier/
│
├── scripts/
│   ├── train_classifier.py
│   ├── evaluate_classifier.py
│   ├── run_gradcam.py
│   └── evaluate_gradcam_iou.py
│
├── src/
│   └── mappillary_classifier/
│       ├── __init__.py
│       ├── datasets.py
│       ├── models.py
│       ├── train.py
│       ├── evaluate.py
│       ├── gradcam_localization.py
│       ├── iou_evaluation.py
│       ├── visualization.py
│       └── utils.py
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Expected dataset format

The training and validation scripts use `torchvision.datasets.ImageFolder`, so the dataset should be arranged like this:

```text
data_root/
├── Train_1000/
│   ├── 0/
│   └── 1/
│
└── Test_new/
    ├── 0/
    └── 1/
```

Class `0` is the negative class and class `1` is the positive class.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

For Grad-CAM functionality:

```bash
pip install grad-cam
```

## Train

```bash
python scripts/train_classifier.py \
    --data-root /path/to/15m \
    --train-folder Train_1000 \
    --valid-folder Test_new \
    --model-name seresnet34 \
    --epochs 15 \
    --batch-size 8 \
    --lr 0.0001 \
    --output-dir /path/to/outputs/models \
    --run-name Resnet34_Bins_size_9x15_15m_1000_SEBlock
```

To use Weights & Biases logging:

```bash
python scripts/train_classifier.py ... --use-wandb --wandb-project AttentionResnet
```

No W&B key is stored in this repository. Use `wandb login` or the `WANDB_API_KEY` environment variable.

## Evaluate

```bash
python scripts/evaluate_classifier.py \
    --data-root /path/to/15m \
    --test-folder Test_new \
    --model-path /path/to/model.pt \
    --model-name resnet34 \
    --batch-size 8 \
    --output-dir /path/to/outputs/evaluation \
    --save-prediction-images
```

## Generate Grad-CAM

```bash
python scripts/run_gradcam.py \
    --image-dir /path/to/predicted_positive_images \
    --model-path /path/to/model.pt \
    --model-name resnet34 \
    --output-dir /path/to/outputs/gradcam \
    --target-class 1
```

## Evaluate Grad-CAM IoU

```bash
python scripts/evaluate_gradcam_iou.py \
    --image-dir /path/to/images \
    --mask-dir /path/to/ground_truth_masks \
    --model-path /path/to/model.pt \
    --model-name resnet34 \
    --output-dir /path/to/outputs/iou \
    --target-class 1 \
    --iou-threshold 0.5
```

## Notes

- The notebook used image normalization `(0.5, 0.5, 0.5)` for both mean and standard deviation. This is kept as the default.
- If your inference scripts use dataset-specific normalization values, train and evaluate with the same values.
- The code supports both full-model checkpoints and `state_dict` checkpoints.
- For public GitHub repositories, `state_dict` checkpoints are preferred.
