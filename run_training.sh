#!/bin/bash
# Training script for Lambda Cloud

# Example usage with your Kaggle-style paths:
# Adjust these paths to match your Lambda Cloud setup

COCO_DIR="/path/to/combined_v4"
IMAGES_DIR="/path/to/combined_v4"

python train_pattern_segmentation.py \
    --coco-dir "$COCO_DIR" \
    --images-dir "$IMAGES_DIR" \
    --batch-size 8 \
    --epochs 22 \
    --lr 1e-4 \
    --num-workers 4 \
    --train-split 0.8 \
    --image-size 512 \
    --ref-size 224 \
    --ref-feature-dim 512 \
    --checkpoint-dir checkpoints \
    --log-dir logs \
    --visualize \
    --num-viz-images 4

# To resume from a checkpoint:
# Add: --resume checkpoints/last.pth
