#!/bin/bash
# Training script for Lambda Cloud
#
# By default the script downloads combined_v4.zip from HuggingFace automatically.
# Set HF_TOKEN env var if the dataset is gated:
#   export HF_TOKEN=hf_...
#
# To use a locally extracted dataset instead, uncomment and set the paths below:
# COCO_DIR="/path/to/combined_v4"
# IMAGES_DIR="/path/to/combined_v4"
# and add:  --coco-dir "$COCO_DIR" --images-dir "$IMAGES_DIR" \

python train_pattern_segmentation.py \
    --hf-repo abshetty/combined_v4 \
    --hf-filename combined_v4.zip \
    --data-dir ./data \
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
