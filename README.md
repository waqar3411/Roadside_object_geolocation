# Roadside object geolocation from street-level images with reduced  supervision.
## The paper can be found [here](https://ieeexplore.ieee.org/abstract/document/10715092)

We propose a method for automated detection and geolocation of roadside objects from street-level images by leveraging historical records of these objects. Such partial and/or noisy geo-records are often held by infrastructure owners and require frequent updating.  We aim to reduce the amount of image-level supervision required for the deployment of deep learning methods to geolocation problem from segmentation masks (very costly) to binary image labels (lower cost). Our proposed method integrates an image classification deep learning pipeline with Grad-CAMs and watershed transform to identify the positions of roadside objects of interest in the images. The geolocation is performed by deploying the existing Markov Random Field-based optimization module. We analyze the robustness of the proposed low-supervision geolocation model to noisy records. We report experiments for the detection of traffic lights and public bins, with geolocation of the latter performed in central Dublin.

## Propsoed Method:
<img width="1262" height="593" alt="Overall_Proposed1" src="https://github.com/user-attachments/assets/e451c7e5-47c8-41fa-9aa6-d0d331a0276c" />




The notebook workflow has been separated into independent components for:

1. Training a ResNet binary classifier.
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
├── Train/
│   ├── 0/
│   └── 1/
│
└── Test/
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
    --data-root /path/to/ \
    --train-folder Train \
    --valid-folder Test \
    --model-name resnet34 \
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
