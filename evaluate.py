#!/usr/bin/env python3
"""
Evaluate trained model on real (non-synthetic) dataset
"""

import argparse
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

# Import from training script
from train_pattern_segmentation import (
    PatternSegmentationNetV3,
    PatternSegmentationDataset,
    calculate_metrics
)


def evaluate_model(checkpoint_path, coco_dir, images_dir, image_size=512, ref_size=224, ref_feature_dim=512):
    """
    Evaluate model on a dataset and print comprehensive metrics.

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        coco_dir: Directory containing COCO JSON files
        images_dir: Directory containing images
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
    print(f"Best validation IoU: {checkpoint.get('best_val_iou', 'unknown'):.4f}")
    print()

    # Create dataset
    print(f"Loading dataset from: {coco_dir}")
    dataset = PatternSegmentationDataset(
        coco_dir,
        images_dir,
        image_size=image_size,
        ref_size=ref_size,
        augment=False  # No augmentation for evaluation
    )
    print(f"Dataset size: {len(dataset)} samples")
    print()

    # Evaluate
    all_ious = []
    all_metrics = []

    print("Calculating metrics for all samples...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Computing metrics"):
            sample = dataset[idx]

            image = sample['image'].unsqueeze(0).to(device)
            reference = sample['reference'].unsqueeze(0).to(device)
            mask = sample['mask'].unsqueeze(0)

            # Predict
            pred = model(image, reference).cpu()

            # Calculate metrics
            metrics = calculate_metrics(pred, mask)

            all_ious.append(metrics['iou'])
            all_metrics.append(metrics)

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Evaluation Results - {Path(coco_dir).name}")
    print(f"{'='*60}")
    print(f"Total samples: {len(dataset)}")
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

    # Return results for further analysis
    return {
        'all_ious': all_ious,
        'all_metrics': all_metrics,
        'mean_iou': np.mean(all_ious),
        'median_iou': np.median(all_ious),
        'dataset_size': len(dataset)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on dataset')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--coco-dir', type=str, required=True,
                        help='Directory containing COCO JSON files')
    parser.add_argument('--images-dir', type=str, required=True,
                        help='Directory containing images')
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

    # Run evaluation
    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        coco_dir=args.coco_dir,
        images_dir=args.images_dir,
        image_size=args.image_size,
        ref_size=args.ref_size,
        ref_feature_dim=args.ref_feature_dim
    )

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
