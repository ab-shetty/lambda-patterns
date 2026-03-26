#!/usr/bin/env python3
"""
Generate visualizations for model predictions on non-synthetic dataset images only,
using Test-Time Augmentation (TTA): averages predictions across original, horizontal
flip, vertical flip, and 90°/180°/270° rotations.
"""

import argparse
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import from training script
from train_pattern_segmentation import (
    PatternSegmentationNetV3,
    PatternSegmentationDataset,
    calculate_metrics
)


def predict_with_tta(model, image, reference):
    """
    Run inference with TTA: original + h-flip + v-flip + 90/180/270 rotations.
    Both image and reference are transformed together (matching training behaviour),
    and each prediction is un-augmented back to the original orientation before averaging.
    """
    preds = []

    # Original
    preds.append(model(image, reference))

    # Horizontal flip
    pred = model(torch.flip(image, [-1]), torch.flip(reference, [-1]))
    preds.append(torch.flip(pred, [-1]))

    # Vertical flip
    pred = model(torch.flip(image, [-2]), torch.flip(reference, [-2]))
    preds.append(torch.flip(pred, [-2]))

    # 90° rotation (k=1)
    pred = model(torch.rot90(image, k=1, dims=[-2, -1]), torch.rot90(reference, k=1, dims=[-2, -1]))
    preds.append(torch.rot90(pred, k=-1, dims=[-2, -1]))

    # 180° rotation (k=2)
    pred = model(torch.rot90(image, k=2, dims=[-2, -1]), torch.rot90(reference, k=2, dims=[-2, -1]))
    preds.append(torch.rot90(pred, k=-2, dims=[-2, -1]))

    # 270° rotation (k=3)
    pred = model(torch.rot90(image, k=3, dims=[-2, -1]), torch.rot90(reference, k=3, dims=[-2, -1]))
    preds.append(torch.rot90(pred, k=-3, dims=[-2, -1]))

    return torch.stack(preds).mean(dim=0)


def visualize_predictions(checkpoint_path, coco_dir, images_dir, output_dir,
                         image_size=512, ref_size=224, ref_feature_dim=512):
    """
    Generate visualization images for non-synthetic samples in dataset using TTA.

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        coco_dir: Directory containing COCO JSON files
        images_dir: Directory containing images
        output_dir: Directory to save visualization images
        image_size: Image size for evaluation
        ref_size: Reference patch size
        ref_feature_dim: Reference feature dimension (must match training)
    """

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model = PatternSegmentationNetV3(in_channels=3, ref_feature_dim=ref_feature_dim)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print()

    # Create dataset
    print(f"Loading dataset from: {coco_dir}")
    dataset = PatternSegmentationDataset(
        coco_dir,
        images_dir,
        image_size=image_size,
        ref_size=ref_size,
        augment=False
    )

    # Filter out synth samples
    original_count = len(dataset.samples)
    dataset.samples = [s for s in dataset.samples if 'synth' not in Path(s['image_path']).name]
    filtered_count = len(dataset.samples)
    print(f"Total samples: {original_count}, after excluding synth: {filtered_count}")
    print(f"TTA passes per sample: 6 (original + h-flip + v-flip + 90°/180°/270°)")
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_path}")
    print()

    # Generate visualizations
    print(f"Generating visualizations for {filtered_count} samples...")

    all_ious = []
    all_metrics = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Visualizing (TTA)"):
            sample = dataset[idx]

            image = sample['image'].unsqueeze(0).to(device)
            reference = sample['reference'].unsqueeze(0).to(device)
            mask = sample['mask']

            # Predict with TTA
            pred = predict_with_tta(model, image, reference).cpu().squeeze()

            # Denormalize for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_vis = sample['image'] * std + mean
            ref_vis = sample['reference'] * std + mean

            # Calculate metrics for this sample
            metrics = calculate_metrics(pred.unsqueeze(0).unsqueeze(0), mask.unsqueeze(0))
            all_ious.append(metrics['iou'])
            all_metrics.append(metrics)

            # Create figure
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            axes[0].imshow(image_vis.permute(1, 2, 0).clip(0, 1))
            axes[0].set_title('Input Image')
            axes[0].axis('off')

            axes[1].imshow(ref_vis.permute(1, 2, 0).clip(0, 1))
            axes[1].set_title('Reference Patch')
            axes[1].axis('off')

            axes[2].imshow(mask.squeeze(), cmap='gray', vmin=0, vmax=1)
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')

            axes[3].imshow(pred, cmap='gray', vmin=0, vmax=1)
            axes[3].set_title(f'Prediction TTA (IoU: {metrics["iou"]:.3f})')
            axes[3].axis('off')

            # Add image filename to figure
            img_name = Path(dataset.samples[idx]['image_path']).name
            fig.suptitle(f'{img_name} - Sample {idx}', fontsize=10)

            plt.tight_layout()
            plt.savefig(output_path / f'sample_{idx:04d}.png', dpi=100, bbox_inches='tight')
            plt.close()

    print(f"\n✓ Saved {filtered_count} visualizations to {output_path}/")

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Evaluation Results (TTA) - {Path(coco_dir).name} (non-synthetic only)")
    print(f"{'='*60}")
    print(f"Total samples: {filtered_count}")
    print()
    print(f"Average IoU:       {np.mean(all_ious):.4f}")
    print(f"Median IoU:        {np.median(all_ious):.4f}")
    print(f"Min IoU:           {np.min(all_ious):.4f}")
    print(f"Max IoU:           {np.max(all_ious):.4f}")
    print(f"Std IoU:           {np.std(all_ious):.4f}")
    print()
    print(f"Average Precision: {np.mean([m['precision'] for m in all_metrics]):.4f}")
    print(f"Average Recall:    {np.mean([m['recall'] for m in all_metrics]):.4f}")
    print(f"Average F1:        {np.mean([m['f1'] for m in all_metrics]):.4f}")

    # Distribution of IoU scores
    print(f"\n{'='*60}")
    print("IoU Distribution:")
    print(f"{'='*60}")
    thresholds = [0.95, 0.9, 0.8, 0.7, 0.5]
    for thresh in thresholds:
        count = sum(iou >= thresh for iou in all_ious)
        pct = 100 * count / len(all_ious)
        print(f"  IoU >= {thresh:.2f}: {count:4d} samples ({pct:5.1f}%)")

    count_low = sum(iou < 0.5 for iou in all_ious)
    pct_low = 100 * count_low / len(all_ious)
    print(f"  IoU <  0.50: {count_low:4d} samples ({pct_low:5.1f}%)")
    print(f"{'='*60}")

    # Percentiles
    print(f"\nIoU Percentiles:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(all_ious, p)
        print(f"  {p:2d}th percentile: {val:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations with TTA (non-synthetic images only)'
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--coco-dir', type=str, required=True,
                        help='Directory containing COCO JSON files')
    parser.add_argument('--images-dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--output-dir', type=str, default='visualizations_real_tta',
                        help='Directory to save visualization images (default: visualizations_real_tta)')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Image size for evaluation (default: 512)')
    parser.add_argument('--ref-size', type=int, default=224,
                        help='Reference patch size (default: 224)')
    parser.add_argument('--ref-feature-dim', type=int, default=512,
                        help='Reference feature dimension (default: 512)')

    args = parser.parse_args()

    # Validate paths
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return

    if not Path(args.coco_dir).exists():
        print(f"Error: COCO directory not found: {args.coco_dir}")
        return

    if not Path(args.images_dir).exists():
        print(f"Error: Images directory not found: {args.images_dir}")
        return

    # Generate visualizations
    visualize_predictions(
        checkpoint_path=args.checkpoint,
        coco_dir=args.coco_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        ref_size=args.ref_size,
        ref_feature_dim=args.ref_feature_dim
    )


if __name__ == "__main__":
    main()
