#!/usr/bin/env python3
"""
Quick setup test script
Verifies that all dependencies are installed and data is accessible.
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('albumentations', 'Albumentations'),
        ('tensorboard', 'TensorBoard'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'TQDM'),
    ]

    failed = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT FOUND")
            failed.append(name)

    if failed:
        print(f"\nMissing packages: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("  All packages found!\n")
    return True


def test_cuda():
    """Test CUDA availability"""
    import torch

    print("Testing CUDA...")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available")
        print(f"  ✓ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  ✓ GPU {i}: {torch.cuda.get_device_name(i)}")

        # Test a simple operation
        try:
            x = torch.randn(100, 100).cuda()
            y = x @ x.T
            print(f"  ✓ CUDA operations working\n")
            return True
        except Exception as e:
            print(f"  ✗ CUDA operations failed: {e}\n")
            return False
    else:
        print("  ✗ CUDA not available - will use CPU (slow!)\n")
        return False


def test_data(coco_dir, images_dir):
    """Test data accessibility"""
    import json

    print(f"Testing data access...")
    coco_path = Path(coco_dir)
    images_path = Path(images_dir)

    if not coco_path.exists():
        print(f"  ✗ COCO directory not found: {coco_dir}")
        return False
    print(f"  ✓ COCO directory exists: {coco_dir}")

    if not images_path.exists():
        print(f"  ✗ Images directory not found: {images_dir}")
        return False
    print(f"  ✓ Images directory exists: {images_dir}")

    # Find COCO files
    coco_files = list(coco_path.glob('*_coco.json'))
    if not coco_files:
        print(f"  ✗ No COCO JSON files found in {coco_dir}")
        return False
    print(f"  ✓ Found {len(coco_files)} COCO files")

    # Check first COCO file
    try:
        with open(coco_files[0], 'r') as f:
            coco_data = json.load(f)
        print(f"  ✓ COCO file format valid")
        print(f"  ✓ Annotations: {len(coco_data.get('annotations', []))}")
        print(f"  ✓ Categories: {len(coco_data.get('categories', []))}")
    except Exception as e:
        print(f"  ✗ Error reading COCO file: {e}")
        return False

    # Check for corresponding image
    image_name = coco_files[0].stem.replace('_coco', '') + '.png'
    image_path = images_path / image_name
    if not image_path.exists():
        print(f"  ✗ Image not found: {image_path}")
        return False
    print(f"  ✓ Corresponding image found: {image_name}\n")

    return True


def test_model_creation():
    """Test if model can be created"""
    import torch

    print("Testing model creation...")
    try:
        # Import model from the training script
        sys.path.insert(0, str(Path(__file__).parent))
        from train_pattern_segmentation import PatternSegmentationNetV3

        model = PatternSegmentationNetV3(in_channels=3, ref_feature_dim=256)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model created successfully")
        print(f"  ✓ Parameters: {num_params:,}\n")

        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        dummy_image = torch.randn(2, 3, 512, 512).to(device)
        dummy_ref = torch.randn(2, 3, 224, 224).to(device)

        with torch.no_grad():
            output = model(dummy_image, dummy_ref)

        print(f"  ✓ Forward pass successful")
        print(f"  ✓ Output shape: {output.shape}\n")

        return True
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test training setup')
    parser.add_argument('--coco-dir', type=str, help='COCO directory to test')
    parser.add_argument('--images-dir', type=str, help='Images directory to test')
    args = parser.parse_args()

    print("=" * 60)
    print("Pattern Segmentation Training - Setup Test")
    print("=" * 60 + "\n")

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False

    # Test CUDA
    if not test_cuda():
        print("WARNING: CUDA not available, training will be very slow!\n")

    # Test data if paths provided
    if args.coco_dir and args.images_dir:
        if not test_data(args.coco_dir, args.images_dir):
            all_passed = False
    else:
        print("Skipping data test (provide --coco-dir and --images-dir to test)\n")

    # Test model
    if not test_model_creation():
        all_passed = False

    # Summary
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed! Ready to train.")
        print("\nRun training with:")
        if args.coco_dir and args.images_dir:
            print(f"  python train_pattern_segmentation.py \\")
            print(f"    --coco-dir {args.coco_dir} \\")
            print(f"    --images-dir {args.images_dir}")
        else:
            print("  ./run_training.sh")
    else:
        print("✗ Some tests failed. Please fix issues before training.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
