#!/usr/bin/env python3
"""
Pattern Segmentation Training Script
Trains a U-Net model with reference-based pattern segmentation using cross-attention and FiLM layers.
"""

import argparse
import json
import random
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt


# ============================================================================
# DATASET DOWNLOAD
# ============================================================================

def download_and_extract_dataset(repo_id='abshetty/combined_v4',
                                  filename='combined_v4.zip',
                                  data_dir='./data'):
    """
    Download and extract the dataset from HuggingFace Hub.
    Returns the path to the extracted dataset directory.
    """
    from huggingface_hub import hf_hub_download

    data_dir = Path(data_dir)
    extract_dir = data_dir / Path(filename).stem  # e.g. ./data/combined_v4

    if extract_dir.exists() and any(extract_dir.glob('*_coco.json')):
        print(f"Dataset already extracted at {extract_dir}, skipping download.")
        return extract_dir

    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {filename} from {repo_id} ...")
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type='dataset',
        local_dir=str(data_dir),
    )
    print(f"Downloaded to {zip_path}")

    print(f"Extracting to {extract_dir} ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(data_dir)

    # The zip may extract into a sub-folder with the same stem name, or directly
    # into data_dir. Detect the actual content directory.
    if not extract_dir.exists():
        # Look for any directory created under data_dir
        subdirs = [p for p in data_dir.iterdir() if p.is_dir()]
        if subdirs:
            extract_dir = subdirs[0]
        else:
            extract_dir = data_dir

    print(f"Dataset ready at {extract_dir}")
    return extract_dir


# ============================================================================
# DATASET UTILITIES
# ============================================================================

def polygon_to_mask(points, height, width):
    """Convert polygon points to binary mask"""
    mask = np.zeros((height, width), dtype=np.uint8)
    points_array = np.array(points, dtype=np.int32).reshape(-1, 2)
    cv2.fillPoly(mask, [points_array], 255)
    return mask


def get_pattern_mask(annotation, height, width):
    """
    Create a binary mask from a COCO annotation's segmentation.
    Handles multiple polygons (outer boundary + holes).
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # First polygon is the outer boundary
    if len(annotation['segmentation']) > 0:
        outer_poly = np.array(annotation['segmentation'][0]).reshape(-1, 2)
        cv2.fillPoly(mask, [outer_poly.astype(np.int32)], 255)

        # Subtract holes (if any)
        for hole_seg in annotation['segmentation'][1:]:
            hole_poly = np.array(hole_seg).reshape(-1, 2)
            cv2.fillPoly(mask, [hole_poly.astype(np.int32)], 0)

    return mask


def sample_random_patch(mask, min_size=128, max_size=512):
    """
    Sample a random rectangular patch that is COMPLETELY inside a pattern mask.
    Returns (x, y, w, h) or None if no valid patch found.
    """
    pattern_coords = np.argwhere(mask > 0)

    if len(pattern_coords) == 0:
        return None

    # Get pattern bounding box
    y_coords, x_coords = pattern_coords[:, 0], pattern_coords[:, 1]
    pattern_x_min, pattern_x_max = x_coords.min(), x_coords.max()
    pattern_y_min, pattern_y_max = y_coords.min(), y_coords.max()
    pattern_width = pattern_x_max - pattern_x_min + 1
    pattern_height = pattern_y_max - pattern_y_min + 1

    # Adjust max size
    max_size = min(max_size, int(pattern_width * 0.7), int(pattern_height * 0.7))
    max_size = max(max_size, min_size)

    # Try random sampling
    for _ in range(200):
        size_range = max_size - min_size
        size_bias = random.random() ** 0.5
        patch_w = int(min_size + size_range * size_bias)
        patch_h = int(min_size + size_range * size_bias)

        center_idx = random.randint(0, len(pattern_coords) - 1)
        center_y, center_x = pattern_coords[center_idx]

        x = center_x - patch_w // 2
        y = center_y - patch_h // 2

        x = max(0, min(x, mask.shape[1] - patch_w))
        y = max(0, min(y, mask.shape[0] - patch_h))

        patch_mask = mask[y:y+patch_h, x:x+patch_w]
        coverage = np.sum(patch_mask > 0) / (patch_w * patch_h)

        if coverage == 1.0:
            return (x, y, patch_w, patch_h)

    # Fallback: try smaller sizes
    for size_fraction in [0.6, 0.5, 0.4, 0.3, 0.2]:
        test_w = max(min_size, int(pattern_width * size_fraction))
        test_h = max(min_size, int(pattern_height * size_fraction))

        for _ in range(50):
            x = random.randint(pattern_x_min, max(pattern_x_min, pattern_x_max - test_w))
            y = random.randint(pattern_y_min, max(pattern_y_min, pattern_y_max - test_h))

            patch_mask = mask[y:y+test_h, x:x+test_w]
            coverage = np.sum(patch_mask > 0) / (test_w * test_h)

            if coverage == 1.0:
                return (x, y, test_w, test_h)

    # Ultimate fallback: tiny box at pattern center
    coords = np.argwhere(mask > 0)
    if len(coords) > 0:
        cy, cx = coords.mean(axis=0).astype(int)
        size = 32
        return (max(0, cx - size//2), max(0, cy - size//2), size, size)

    return (0, 0, 32, 32)


class PatternSegmentationDataset(Dataset):
    """
    Dataset for pattern segmentation training.
    Loads COCO format annotations and generates training pairs.
    """
    def __init__(self, coco_dir, images_dir, image_size=512, ref_size=224,
                 min_patch_size=128, max_patch_size=512, augment=True):
        """
        Args:
            coco_dir: Directory containing COCO JSON files
            images_dir: Directory containing original images
            image_size: Target size for full images (will be resized)
            ref_size: Target size for reference patches (224 for ResNet)
            min_patch_size: Minimum size for reference patch sampling
            max_patch_size: Maximum size for reference patch sampling
            augment: Whether to apply data augmentation
        """
        self.coco_dir = Path(coco_dir)
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.ref_size = ref_size
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.augment = augment

        # Load all COCO files
        self.coco_files = sorted(list(self.coco_dir.glob('*_coco.json')))

        if len(self.coco_files) == 0:
            raise ValueError(f"No COCO files found in {coco_dir}")

        # Build dataset: one sample per category per image
        self.samples = []
        self._build_dataset()

        print(f"Loaded {len(self.samples)} training samples from {len(self.coco_files)} images")

        # Augmentation transforms
        if augment:
            # ColorJitter applied to full image before reference extraction so
            # both image and reference naturally share the same color shift
            self.color_jitter = A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.2,
                hue=0.1,
                p=0.5
            )
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=1.0),
            ])
        else:
            self.color_jitter = None
            self.transform = None

        # Normalization (ImageNet stats)
        self.normalize = A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _build_dataset(self):
        """Build list of training samples"""
        for coco_file in self.coco_files:
            # Load COCO data
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)

            # Get image path
            image_name = coco_file.stem.replace('_coco', '') + '.png'
            image_path = self.images_dir / image_name

            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}, skipping...")
                continue

            # Group annotations by category
            category_groups = {}
            for ann in coco_data['annotations']:
                cat_id = ann['category_id']
                if cat_id not in category_groups:
                    category_groups[cat_id] = []
                category_groups[cat_id].append(ann)

            # Create one sample per category
            for cat_id, annotations in category_groups.items():
                self.samples.append({
                    'image_path': str(image_path),
                    'annotations': annotations,
                    'category_id': cat_id,
                    'image_height': coco_data['images'][0]['height'],
                    'image_width': coco_data['images'][0]['width']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            image: Full image [3, image_size, image_size]
            reference: Reference patch [3, ref_size, ref_size]
            mask: Target segmentation mask [1, image_size, image_size]
        """
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Create combined target mask for this category
        target_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        instance_masks = []

        for ann in sample['annotations']:
            instance_mask = get_pattern_mask(ann, orig_h, orig_w)
            target_mask = cv2.bitwise_or(target_mask, instance_mask)
            instance_masks.append(instance_mask)

        # Randomly select one instance to sample reference from
        selected_mask = random.choice(instance_masks)

        # Sample reference patch bbox
        patch_bbox = sample_random_patch(selected_mask, self.min_patch_size, self.max_patch_size)

        # Apply color jitter to full image before extracting reference patch
        # so both naturally share the same color shift
        if self.color_jitter is not None:
            image = self.color_jitter(image=image)['image']

        # Extract reference patch after color jitter, before spatial augmentation
        x, y, w, h = patch_bbox
        reference_patch = image[y:y+h, x:x+w].copy()

        # Apply SAME augmentation to image, mask, and reference patch
        if self.transform is not None:
            seed = np.random.randint(0, 2**32 - 1)

            np.random.seed(seed)
            random.seed(seed)
            transformed = self.transform(image=image, mask=target_mask)
            image_aug = transformed['image']
            mask_aug = transformed['mask']

            np.random.seed(seed)
            random.seed(seed)
            transformed_ref = self.transform(image=reference_patch)
            reference_aug = transformed_ref['image']
        else:
            image_aug = image
            mask_aug = target_mask
            reference_aug = reference_patch

        # Resize image and mask
        image_resized = cv2.resize(image_aug, (self.image_size, self.image_size))
        mask_resized = cv2.resize(mask_aug, (self.image_size, self.image_size))

        # Resize reference patch
        reference_resized = cv2.resize(reference_aug, (self.ref_size, self.ref_size))

        # Normalize
        image_norm = self.normalize(image=image_resized)['image']
        reference_norm = self.normalize(image=reference_resized)['image']

        # Convert to tensors
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).float()
        reference_tensor = torch.from_numpy(reference_norm).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float() / 255.0

        return {
            'image': image_tensor,
            'reference': reference_tensor,
            'mask': mask_tensor
        }


def create_dataloaders(coco_dir, images_dir, batch_size=4, num_workers=4,
                       train_split=0.8, image_size=512, ref_size=224):
    """
    Create train and validation dataloaders with proper splitting.
    """
    # Get all COCO files
    coco_dir = Path(coco_dir)
    all_coco_files = sorted(list(coco_dir.glob('*_coco.json')))

    # Split into train/val
    n_train = int(len(all_coco_files) * train_split)
    train_files = all_coco_files[:n_train]
    val_files = all_coco_files[n_train:]

    print(f"Total images: {len(all_coco_files)}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Create temporary directories with symlinks for split
    temp_dir = Path(tempfile.mkdtemp())
    train_coco_dir = temp_dir / 'train_coco'
    val_coco_dir = temp_dir / 'val_coco'
    train_coco_dir.mkdir()
    val_coco_dir.mkdir()

    # Copy files to temp directories
    for f in train_files:
        shutil.copy(f, train_coco_dir / f.name)
    for f in val_files:
        shutil.copy(f, val_coco_dir / f.name)

    # Create datasets
    train_dataset = PatternSegmentationDataset(
        train_coco_dir, images_dir,
        image_size=image_size, ref_size=ref_size, augment=True
    )

    val_dataset = PatternSegmentationDataset(
        val_coco_dir, images_dir,
        image_size=image_size, ref_size=ref_size, augment=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, temp_dir


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ReferenceEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()

        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.stem    = nn.Sequential(*list(resnet.children())[:4])
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4

        self.proj1 = nn.Conv2d(256,  feature_dim, 1)
        self.proj2 = nn.Conv2d(512,  feature_dim, 1)
        self.proj3 = nn.Conv2d(1024, feature_dim, 1)
        self.proj4 = nn.Conv2d(2048, feature_dim, 1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.feature_dim = feature_dim

    def forward(self, x):
        x  = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        ref_spatial = {
            'scale1': self.proj1(f1),
            'scale2': self.proj2(f2),
            'scale3': self.proj3(f3),
            'scale4': self.proj4(f4),
        }

        global_feat = self.gap(f4).flatten(1)
        ref_global = self.global_proj(global_feat)

        return ref_spatial, ref_global


class CrossAttentionBlock(nn.Module):
    def __init__(self, image_channels, ref_channels, num_heads=4):
        super().__init__()

        self.attn_dim = max(image_channels, ref_channels)

        self.q_proj = nn.Conv2d(image_channels, self.attn_dim, 1)
        self.k_proj = nn.Conv2d(ref_channels,   self.attn_dim, 1)
        self.v_proj = nn.Conv2d(ref_channels,   self.attn_dim, 1)

        self.num_heads = num_heads
        self.head_dim  = self.attn_dim // num_heads
        assert self.attn_dim % num_heads == 0

        self.out_proj = nn.Sequential(
            nn.Conv2d(self.attn_dim, image_channels, 1),
            nn.BatchNorm2d(image_channels),
            nn.ReLU(inplace=True)
        )

        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(0.1)

    def forward(self, image_feat, ref_feat):
        B, C, H, W = image_feat.shape
        _, _, h, w = ref_feat.shape

        Q = self.q_proj(image_feat)
        K = self.k_proj(ref_feat)
        V = self.v_proj(ref_feat)

        nh = self.num_heads
        hd = self.head_dim

        Q = Q.view(B, nh, hd, H*W).permute(0, 1, 3, 2)
        K = K.view(B, nh, hd, h*w).permute(0, 1, 3, 2)
        V = V.view(B, nh, hd, h*w).permute(0, 1, 3, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, self.attn_dim, H, W)
        out = self.out_proj(out) + image_feat

        return out


class FiLMLayer(nn.Module):
    def __init__(self, in_channels, ref_feature_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(ref_feature_dim, in_channels)
        self.beta_fc  = nn.Linear(ref_feature_dim, in_channels)
        nn.init.ones_(self.gamma_fc.bias)

    def forward(self, x, ref_features):
        gamma = self.gamma_fc(ref_features).unsqueeze(-1).unsqueeze(-1)
        beta  = self.beta_fc(ref_features).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ref_channels, ref_feature_dim,
                 use_cross_attn=True, num_heads=4):
        super().__init__()

        self.use_cross_attn = use_cross_attn

        self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)

        if use_cross_attn:
            self.ref_module = CrossAttentionBlock(out_channels, ref_channels, num_heads=num_heads)
        else:
            self.ref_module = FiLMLayer(out_channels, ref_feature_dim)

    def forward(self, x, skip, ref_spatial, ref_global):
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        if self.use_cross_attn:
            x = self.ref_module(x, ref_spatial)
        else:
            x = self.ref_module(x, ref_global)

        return x


class PatternSegmentationNetV3(nn.Module):
    """
    U-Net with Hybrid Cross-Attention / FiLM Reference Injection.
    """
    def __init__(self, in_channels=3, ref_feature_dim=256):
        super().__init__()

        self.ref_feature_dim = ref_feature_dim

        self.reference_encoder = ReferenceEncoder(feature_dim=ref_feature_dim)

        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        self.bottleneck      = DoubleConv(512, 1024)
        self.bottleneck_film = FiLMLayer(1024, ref_feature_dim)

        self.dec4 = DecoderBlock(1024, 512, ref_channels=ref_feature_dim,
                                 ref_feature_dim=ref_feature_dim, use_cross_attn=True,  num_heads=4)
        self.dec3 = DecoderBlock(512,  256, ref_channels=ref_feature_dim,
                                 ref_feature_dim=ref_feature_dim, use_cross_attn=True,  num_heads=4)
        self.dec2 = DecoderBlock(256,  128, ref_channels=ref_feature_dim,
                                 ref_feature_dim=ref_feature_dim, use_cross_attn=True,  num_heads=4)
        self.dec1 = DecoderBlock(128,   64, ref_channels=ref_feature_dim,
                                 ref_feature_dim=ref_feature_dim, use_cross_attn=False, num_heads=4)

        self.out_conv = nn.Conv2d(64, 1, 1)

    def forward(self, image, reference_patch):
        ref_spatial, ref_global = self.reference_encoder(reference_patch)

        x1, skip1 = self.enc1(image)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        x4, skip4 = self.enc4(x3)

        x = self.bottleneck(x4)
        x = self.bottleneck_film(x, ref_global)

        x = self.dec4(x, skip4, ref_spatial['scale4'], ref_global)
        x = self.dec3(x, skip3, ref_spatial['scale3'], ref_global)
        x = self.dec2(x, skip2, F.adaptive_avg_pool2d(ref_spatial['scale1'], (28, 28)), ref_global)
        x = self.dec1(x, skip1, ref_spatial['scale1'], ref_global)

        x = self.out_conv(x)
        return torch.sigmoid(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.flatten(1)
        target = target.flatten(1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combination of BCE and Dice loss."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union (IoU)."""
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def calculate_metrics(pred, target, threshold=0.5):
    """Calculate various segmentation metrics."""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    # True Positives, False Positives, False Negatives
    tp = (pred_binary * target_binary).sum(dim=(1, 2, 3))
    fp = (pred_binary * (1 - target_binary)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_binary) * target_binary).sum(dim=(1, 2, 3))

    # Precision and Recall
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    # IoU
    iou = calculate_iou(pred, target, threshold)

    return {
        'iou': iou,
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1': f1.mean().item()
    }


class Trainer:
    """Trainer class for pattern segmentation model."""
    def __init__(self, model, train_loader, val_loader, device,
                 lr=1e-4, checkpoint_dir='checkpoints', log_dir='logs', step_size=8):
        self.model = model.to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss and optimizer
        self.criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

        # Get base model (unwrap DataParallel if needed)
        base_model = model.module if isinstance(model, nn.DataParallel) else model

        self.optimizer = optim.Adam([
            {'params': base_model.reference_encoder.parameters(), 'lr': 1e-5},
            {'params': [p for n, p in base_model.named_parameters()
                        if 'reference_encoder' not in n and 'ref_module' not in n], 'lr': 1e-4},
            {'params': [p for n, p in base_model.named_parameters()
                        if 'ref_module' in n], 'lr': 5e-5},
        ])
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=0.2)

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Logging
        self.writer = SummaryWriter(log_dir)

        # AMP scaler
        self.scaler = torch.amp.GradScaler('cuda')

        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0
        self.epoch = 0

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        total_iou = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch} [Train]')
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            references = batch['reference'].to(self.device)
            masks = batch['mask'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = self.model(images, references)
            loss = self.criterion(outputs.float(), masks)

            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Calculate metrics
            with torch.no_grad():
                iou = calculate_iou(outputs, masks)

            # Update stats
            total_loss += loss.item()
            total_iou += iou

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}'
            })

            # Log to tensorboard
            global_step = self.epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/IoU', iou, global_step)

        avg_loss = total_loss / len(self.train_loader)
        avg_iou = total_iou / len(self.train_loader)

        return avg_loss, avg_iou

    def validate(self):
        """Validate the model"""
        self.model.eval()

        total_loss = 0
        total_metrics = {
            'iou': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0
        }

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.epoch} [Val]')
            for batch in pbar:
                # Move to device
                images = batch['image'].to(self.device)
                references = batch['reference'].to(self.device)
                masks = batch['mask'].to(self.device)

                # Forward pass
                outputs = self.model(images, references)

                # Calculate loss
                loss = self.criterion(outputs, masks)

                # Calculate metrics
                metrics = calculate_metrics(outputs, masks)

                # Update stats
                total_loss += loss.item()
                for key in total_metrics:
                    total_metrics[key] += metrics[key]

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'iou': f'{metrics["iou"]:.4f}'
                })

        avg_loss = total_loss / len(self.val_loader)
        for key in total_metrics:
            total_metrics[key] /= len(self.val_loader)

        return avg_loss, total_metrics

    def save_checkpoint(self, filename):
        # Handle DataParallel wrapper
        model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_iou': self.best_val_iou
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        # Load into the underlying model (unwrap DataParallel if needed)
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_iou = checkpoint['best_val_iou']

    def train(self, num_epochs):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")

        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()

            # Train
            train_loss, train_iou = self.train_epoch()

            # Validate
            val_loss, val_metrics = self.validate()

            # Learning rate scheduling
            self.scheduler.step()

            # Log to tensorboard
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Train_IoU', train_iou, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Epoch/Val_{key.upper()}', value, epoch)
            self.writer.add_scalar('Epoch/LR', self.optimizer.param_groups[0]['lr'], epoch)

            epoch_time = time.time() - start_time

            # Print summary
            print(f"\nEpoch {epoch}/{num_epochs-1}")
            print(f"  Time: {epoch_time:.1f}s")
            print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_metrics['iou']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Save checkpoints
            self.save_checkpoint('last.pth')

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

            if val_metrics['iou'] > self.best_val_iou:
                self.best_val_iou = val_metrics['iou']
                self.save_checkpoint('best_iou.pth')
                print(f"  → Saved best IoU checkpoint")

        print(f"\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation IoU: {self.best_val_iou:.4f}")

        self.writer.close()


def visualize_predictions(model, dataset, device, num_images=4, output_path='predictions.png'):
    """Visualize model predictions on sample data."""
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    actual_model.eval()

    # Group samples by image path
    image_groups = {}
    for idx, sample_info in enumerate(dataset.samples):
        img_path = sample_info['image_path']
        if img_path not in image_groups:
            image_groups[img_path] = []
        image_groups[img_path].append(idx)

    # Select random images
    selected_images = random.sample(list(image_groups.keys()),
                                   min(num_images, len(image_groups)))

    # Pick one random sample from each selected image
    selected_indices = []
    for img_path in selected_images:
        idx = random.choice(image_groups[img_path])
        selected_indices.append(idx)

    num_samples = len(selected_indices)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i, idx in enumerate(selected_indices):
            sample = dataset[idx]

            image = sample['image'].unsqueeze(0).to(device)
            reference = sample['reference'].unsqueeze(0).to(device)
            mask = sample['mask']

            # Predict
            pred = actual_model(image, reference).cpu().squeeze()

            # Denormalize for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_vis = sample['image'] * std + mean
            ref_vis = sample['reference'] * std + mean

            # Get image filename for title
            img_name = Path(dataset.samples[idx]['image_path']).name

            # Plot
            axes[i, 0].imshow(image_vis.permute(1, 2, 0).clip(0, 1))
            axes[i, 0].set_title(f'Image: {img_name}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(ref_vis.permute(1, 2, 0).clip(0, 1))
            axes[i, 1].set_title('Reference Patch')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(mask.squeeze(), cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title('Ground Truth')
            axes[i, 2].axis('off')

            axes[i, 3].imshow(pred, cmap='gray', vmin=0, vmax=1)
            axes[i, 3].set_title('Prediction')
            axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved predictions to {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train Pattern Segmentation Model')

    # Data paths (auto-downloaded from HuggingFace if not provided)
    parser.add_argument('--coco-dir', type=str, default=None,
                        help='Directory containing COCO JSON files (overrides HuggingFace download)')
    parser.add_argument('--images-dir', type=str, default=None,
                        help='Directory containing images (overrides HuggingFace download)')

    # HuggingFace dataset download
    parser.add_argument('--hf-repo', type=str, default='abshetty/combined_v4',
                        help='HuggingFace dataset repo ID (default: abshetty/combined_v4)')
    parser.add_argument('--hf-filename', type=str, default='combined_v4.zip',
                        help='Zip filename in the HuggingFace repo (default: combined_v4.zip)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Local directory to download/extract dataset into (default: ./data)')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=22,
                        help='Number of training epochs (default: 22)')
    parser.add_argument('--step-size', type=int, default=8,
                        help='LR scheduler step size in epochs (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of data loading workers (default: 2)')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Train/val split ratio (default: 0.8)')

    # Model parameters
    parser.add_argument('--image-size', type=int, default=512,
                        help='Image size for training (default: 512)')
    parser.add_argument('--ref-size', type=int, default=224,
                        help='Reference patch size (default: 224)')
    parser.add_argument('--ref-feature-dim', type=int, default=512,
                        help='Reference feature dimension (default: 512)')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for tensorboard logs (default: logs)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization after training')
    parser.add_argument('--num-viz-images', type=int, default=4,
                        help='Number of images to visualize (default: 4)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPUs: {torch.cuda.device_count()}")

    # Resolve data directories (download from HuggingFace if not provided locally)
    if args.coco_dir and args.images_dir:
        coco_dir = args.coco_dir
        images_dir = args.images_dir
    else:
        dataset_dir = download_and_extract_dataset(
            repo_id=args.hf_repo,
            filename=args.hf_filename,
            data_dir=args.data_dir,
        )
        coco_dir = str(dataset_dir)
        images_dir = str(dataset_dir)

    # Create dataloaders
    print("\nPreparing data...")
    train_loader, val_loader, temp_dir = create_dataloaders(
        coco_dir, images_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        image_size=args.image_size,
        ref_size=args.ref_size
    )

    # Create model
    print("\nCreating model...")
    model = PatternSegmentationNetV3(in_channels=3, ref_feature_dim=args.ref_feature_dim)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = Trainer(
        model, train_loader, val_loader, device,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        step_size=args.step_size
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {trainer.epoch}")

    # Train
    try:
        trainer.train(args.epochs)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint('interrupted.pth')

    # Visualize results
    if args.visualize:
        print("\nGenerating predictions visualization...")
        visualize_predictions(
            trainer.model, val_loader.dataset, device,
            num_images=args.num_viz_images,
            output_path='predictions.png'
        )

    # Cleanup
    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_dir)
    print("Done!")


if __name__ == "__main__":
    main()
