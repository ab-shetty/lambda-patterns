# Pattern Segmentation Training on Lambda Cloud

This guide will help you set up and run the pattern segmentation training on a Lambda Cloud GPU instance.

## Prerequisites

- Lambda Cloud GPU instance (recommended: A100 or H100 for multi-GPU training)
- Your dataset uploaded to the instance

## Setup Instructions

### 1. Install Dependencies

```bash
# Update system
sudo apt-get update

# Install Python dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python-headless numpy albumentations tensorboard matplotlib tqdm
```

### 2. Upload Your Data

Transfer your dataset to the Lambda Cloud instance. You can use `scp`, `rsync`, or Lambda's web interface.

```bash
# Example using scp from your local machine:
scp -r /path/to/combined_v4 user@lambda-instance:/home/ubuntu/data/
```

### 3. Upload Training Scripts

```bash
# Upload the training script
scp train_pattern_segmentation.py user@lambda-instance:/home/ubuntu/
scp run_training.sh user@lambda-instance:/home/ubuntu/
```

### 4. Configure Paths

Edit `run_training.sh` to set your data paths:

```bash
nano run_training.sh
# Update COCO_DIR and IMAGES_DIR to match your setup
```

### 5. Run Training

```bash
# Make script executable
chmod +x run_training.sh

# Run training
./run_training.sh
```

## Command-Line Options

```bash
python train_pattern_segmentation.py --help
```

### Key Arguments:

- `--coco-dir`: Directory containing COCO JSON annotation files (required)
- `--images-dir`: Directory containing PNG images (required)
- `--batch-size`: Batch size (default: 4, increase for more GPUs)
- `--epochs`: Number of training epochs (default: 22)
- `--lr`: Learning rate (default: 1e-4)
- `--num-workers`: Data loading workers (default: 2)
- `--train-split`: Train/validation split ratio (default: 0.8)
- `--image-size`: Input image size (default: 512)
- `--ref-size`: Reference patch size (default: 224)
- `--ref-feature-dim`: Feature dimension (default: 512)
- `--checkpoint-dir`: Where to save checkpoints (default: checkpoints)
- `--log-dir`: TensorBoard log directory (default: logs)
- `--resume`: Resume from checkpoint (e.g., checkpoints/last.pth)
- `--visualize`: Generate prediction visualizations after training
- `--num-viz-images`: Number of images to visualize (default: 4)

## Example Commands

### Basic Training
```bash
python train_pattern_segmentation.py \
    --coco-dir /home/ubuntu/data/combined_v4 \
    --images-dir /home/ubuntu/data/combined_v4 \
    --batch-size 8 \
    --epochs 22
```

### Multi-GPU Training (Automatic)
The script automatically detects and uses all available GPUs with DataParallel.

```bash
# For 2x A100 (80GB each), you can use larger batch size:
python train_pattern_segmentation.py \
    --coco-dir /home/ubuntu/data/combined_v4 \
    --images-dir /home/ubuntu/data/combined_v4 \
    --batch-size 16 \
    --epochs 22 \
    --num-workers 8
```

### Resume Training
```bash
python train_pattern_segmentation.py \
    --coco-dir /home/ubuntu/data/combined_v4 \
    --images-dir /home/ubuntu/data/combined_v4 \
    --resume checkpoints/last.pth
```

### Training with Visualization
```bash
python train_pattern_segmentation.py \
    --coco-dir /home/ubuntu/data/combined_v4 \
    --images-dir /home/ubuntu/data/combined_v4 \
    --batch-size 8 \
    --epochs 22 \
    --visualize \
    --num-viz-images 8
```

## Monitoring Training

### TensorBoard

```bash
# In a separate terminal or tmux session:
tensorboard --logdir logs --port 6006

# Then access via SSH tunnel from your local machine:
# ssh -L 6006:localhost:6006 user@lambda-instance
# Open browser: http://localhost:6006
```

### Training Progress

The script prints progress to stdout:
- Per-batch loss and IoU during training
- Epoch summaries with train/val metrics
- Learning rate updates
- Checkpoint saves

## Output Files

### Checkpoints
- `checkpoints/last.pth`: Latest checkpoint (saved every epoch)
- `checkpoints/best_iou.pth`: Best validation IoU checkpoint
- `checkpoints/interrupted.pth`: Saved if training is interrupted (Ctrl+C)

### Logs
- `logs/`: TensorBoard logs with training curves

### Visualizations
- `predictions.png`: Sample predictions (if --visualize flag is used)

## Tips for Lambda Cloud

### 1. Use tmux/screen
Training can take hours, so use tmux or screen to prevent disconnection:

```bash
# Start tmux session
tmux new -s training

# Run training
./run_training.sh

# Detach: Ctrl+B then D
# Reattach: tmux attach -t training
```

### 2. Monitor GPU Usage
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi
```

### 3. Optimize Batch Size
- Single A100 (40GB): batch_size 8-12
- Single A100 (80GB): batch_size 16-24
- 2x A100 (80GB): batch_size 32-48

Start with smaller batch sizes and increase until you hit OOM errors.

### 4. Save Checkpoints Regularly
The script saves `last.pth` every epoch automatically. If training crashes, resume with:

```bash
python train_pattern_segmentation.py \
    --coco-dir /path/to/data \
    --images-dir /path/to/data \
    --resume checkpoints/last.pth
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch-size`
- Reduce `--image-size` (e.g., 384 or 256)
- Reduce `--num-workers`

### Slow Data Loading
- Increase `--num-workers` (typically 2x or 4x number of GPUs)
- Ensure data is on fast local SSD, not network storage

### Poor Performance
- Check TensorBoard logs for learning curves
- Ensure augmentation is working (visualize training samples)
- Try adjusting learning rate or scheduler parameters

## Performance Expectations

### Training Speed (approximate)
- Single A100 (80GB), batch_size=16: ~45 sec/epoch
- 2x A100 (80GB), batch_size=32: ~30 sec/epoch
- 4x A100 (80GB), batch_size=64: ~20 sec/epoch

### Expected Metrics
After 22 epochs, you should see:
- Validation IoU: ~0.85-0.95 (dataset dependent)
- Validation F1: ~0.90-0.97
- Training loss: <0.1

## Advanced: Hyperparameter Tuning

Key hyperparameters to tune:
1. `--ref-feature-dim`: 256 (faster) vs 512 (better quality)
2. Learning rate schedule: Modify `step_size` and `gamma` in the code
3. Loss weights: Adjust BCE vs Dice weights in `CombinedLoss`
4. Augmentation: Modify `PatternSegmentationDataset.transform`

## Contact & Support

For issues or questions about the script, check:
1. TensorBoard logs for training issues
2. Console output for data loading errors
3. GPU memory with `nvidia-smi`
