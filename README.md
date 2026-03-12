# Pattern Segmentation Training

Deep learning pipeline for training a U-Net model with reference-based pattern segmentation using cross-attention and FiLM layers.

## Overview

This repository contains a production-ready training pipeline for pattern segmentation, designed to run on Lambda Cloud GPU instances. The model uses a hybrid architecture combining:

- **U-Net encoder-decoder** for segmentation
- **ResNet50 reference encoder** for pattern feature extraction
- **Cross-attention blocks** for spatial pattern matching
- **FiLM layers** for global pattern conditioning

## Features

- ✅ Multi-GPU training (automatic DataParallel)
- ✅ COCO format dataset support
- ✅ Comprehensive data augmentation
- ✅ TensorBoard logging
- ✅ Automatic checkpointing
- ✅ Resume from checkpoint
- ✅ Flexible command-line interface
- ✅ Visualization utilities

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Your Setup

```bash
python test_setup.py --coco-dir /path/to/data --images-dir /path/to/data
```

### 3. Configure Training

Edit `run_training.sh` to set your data paths:

```bash
nano run_training.sh
```

### 4. Start Training

```bash
./run_training.sh
```

## Usage

### Basic Training

```bash
python train_pattern_segmentation.py \
    --coco-dir /path/to/coco/annotations \
    --images-dir /path/to/images \
    --batch-size 8 \
    --epochs 22
```

### Multi-GPU Training

The script automatically detects and uses all available GPUs:

```bash
python train_pattern_segmentation.py \
    --coco-dir /path/to/data \
    --images-dir /path/to/data \
    --batch-size 16 \
    --num-workers 8
```

### Resume Training

```bash
python train_pattern_segmentation.py \
    --coco-dir /path/to/data \
    --images-dir /path/to/data \
    --resume checkpoints/last.pth
```

### With Visualization

```bash
python train_pattern_segmentation.py \
    --coco-dir /path/to/data \
    --images-dir /path/to/data \
    --visualize \
    --num-viz-images 8
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--coco-dir` | Required | Directory containing COCO JSON files |
| `--images-dir` | Required | Directory containing images |
| `--batch-size` | 4 | Batch size for training |
| `--epochs` | 22 | Number of training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--num-workers` | 2 | Number of data loading workers |
| `--train-split` | 0.8 | Train/validation split ratio |
| `--image-size` | 512 | Input image size |
| `--ref-size` | 224 | Reference patch size |
| `--ref-feature-dim` | 512 | Reference feature dimension |
| `--checkpoint-dir` | checkpoints | Checkpoint save directory |
| `--log-dir` | logs | TensorBoard log directory |
| `--resume` | None | Path to checkpoint to resume from |
| `--visualize` | False | Generate visualizations after training |
| `--num-viz-images` | 4 | Number of images to visualize |

## Model Architecture

### PatternSegmentationNetV3

- **Input**: Full image (512×512) + Reference patch (224×224)
- **Output**: Segmentation mask (512×512)
- **Encoder**: U-Net style with 4 downsampling blocks
- **Reference Encoder**: ResNet50 with multi-scale feature extraction
- **Decoder**: 4 upsampling blocks with hybrid attention
  - Bottleneck (16×16): FiLM conditioning
  - Dec4 (32×32): Cross-attention with 7×7 ref features
  - Dec3 (64×64): Cross-attention with 14×14 ref features
  - Dec2 (128×128): Cross-attention with 56×56 ref features
  - Dec1 (256×256): FiLM conditioning

**Parameters**: ~70M (with ref_feature_dim=512)

## Dataset Format

The training script expects COCO format annotations:

```
data/
├── image1.png
├── image1_coco.json
├── image2.png
├── image2_coco.json
└── ...
```

Each COCO file should contain:
- `images`: Image metadata
- `annotations`: Segmentation annotations with polygon coordinates
- `categories`: Pattern categories

## Outputs

### Checkpoints
- `checkpoints/last.pth`: Latest checkpoint (every epoch)
- `checkpoints/best_iou.pth`: Best validation IoU
- `checkpoints/interrupted.pth`: Saved on Ctrl+C

### Logs
- `logs/`: TensorBoard event files

### Visualizations
- `predictions.png`: Sample predictions (if `--visualize`)

## Monitoring

### TensorBoard

```bash
tensorboard --logdir logs --port 6006
```

Then access via SSH tunnel:
```bash
ssh -L 6006:localhost:6006 user@lambda-instance
```

Open browser: http://localhost:6006

### Metrics Tracked

- Training: Loss, IoU
- Validation: Loss, IoU, Precision, Recall, F1
- Learning rate per epoch

## Performance

### Training Speed (approximate)

| Hardware | Batch Size | Time/Epoch |
|----------|------------|------------|
| 1× A100 (80GB) | 16 | ~45s |
| 2× A100 (80GB) | 32 | ~30s |
| 4× A100 (80GB) | 64 | ~20s |

### Expected Results

After 22 epochs:
- Validation IoU: 0.85-0.95
- Validation F1: 0.90-0.97
- Training Loss: <0.1

## Lambda Cloud Setup

See [LAMBDA_SETUP.md](LAMBDA_SETUP.md) for detailed instructions on:
- Setting up a Lambda Cloud instance
- Installing dependencies
- Transferring data
- Optimizing for multi-GPU training
- Troubleshooting common issues

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 16GB+ GPU memory (for batch_size=8, image_size=512)

See `requirements.txt` for complete package list.

## Troubleshooting

### Out of Memory
- Reduce `--batch-size`
- Reduce `--image-size` (try 384 or 256)
- Reduce `--ref-feature-dim` (try 256)

### Slow Training
- Increase `--num-workers`
- Use faster storage (local SSD vs network)
- Increase `--batch-size` if GPU memory allows

### Poor Convergence
- Check data augmentation (samples should vary)
- Verify annotations are correct
- Try adjusting learning rate
- Increase training epochs

## License

MIT

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pattern_segmentation_2024,
  title={Pattern Segmentation Training Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/ab-shetty/lambda-patterns}
}
```
