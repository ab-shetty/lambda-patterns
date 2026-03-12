# Model Improvement Strategies

Current performance: **0.8993 validation IoU** (89.93%)
Real-world performance: **0.75 median IoU** on real PDFs

This document outlines strategies to push performance higher, leveraging the GH200's massive compute capabilities.

---

## Quick Wins (Minimal Code Changes)

### 1. **Increase Batch Size** ⭐ (Easiest, Likely +2-3% IoU)

**Current:** `batch_size=4` (default)
**Available:** 480GB GPU memory
**Recommended:** `batch_size=64` or even `128`

```bash
python train_pattern_segmentation.py \
    --coco-dir ~/combined_v4 \
    --images-dir ~/combined_v4 \
    --batch-size 64 \
    --epochs 30 \
    --ref-feature-dim 512
```

**Why it works:**
- Larger batches = better gradient estimates
- More stable batch normalization statistics
- Better convergence to global minimum
- GH200's massive memory is underutilized at batch_size=4

**Expected improvement:** 0.90 → **0.92 IoU**

---

### 2. **Better Learning Rate Schedule**

**Current:** StepLR with `gamma=0.2` every 8 epochs
**Recommended:** Cosine Annealing with Warmup

**Benefits:**
- Smooth learning rate decay (vs sharp steps)
- Warmup prevents early overfitting
- Better final convergence

**Implementation:**
```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Warmup for 5 epochs
warmup_scheduler = LinearLR(
    optimizer, start_factor=0.1, total_iters=5
)

# Cosine decay for remaining epochs
cosine_scheduler = CosineAnnealingLR(
    optimizer, T_max=35, eta_min=1e-7
)

# Combine
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[5]
)
```

**Expected improvement:** +1-2% IoU → **0.91-0.92 IoU**

---

### 3. **Train Longer**

**Current:** 22 epochs
**Recommended:** 30-40 epochs

With cosine annealing, the model continues improving as LR decays smoothly to near-zero. Training stopped at epoch 22 but loss was still decreasing.

**Expected improvement:** +0.5-1% IoU

---

## Medium Effort (Code Modifications)

### 4. **Generate More Synthetic Training Data** ⭐ (High ROI)

**Current:** 3028 images in combined_v4
**Recommended:** 10,000-30,000 images

**Why it works:**
- More data = better generalization
- Synthetic data is cheap to generate
- Can target failure cases (hard negatives)
- Increases pattern diversity

**Strategies:**

**a) Increase Dataset Size (10x)**
- Generate 30,000 synthetic images
- Vary: patterns, layouts, colors, noise levels
- Include edge cases: overlapping patterns, partial patterns, distortions

**b) Targeted Data Generation**
- Analyze failure cases from real PDF evaluation
- Generate synthetic data that mimics those failures
- Examples: dense patterns, small patterns, low contrast

**c) Data Quality Improvements**
- More realistic PDF rendering
- Better pattern variation
- Authentic noise/artifacts
- Varied fonts and layouts

**Implementation:**
```bash
# Generate 10x more data with your existing pipeline
# Then retrain:
python train_pattern_segmentation.py \
    --coco-dir ~/combined_v10 \
    --images-dir ~/combined_v10 \
    --batch-size 64 \
    --epochs 50
```

**Expected improvement:** +3-5% IoU → **0.93-0.95 IoU**

**Bonus:** Better real-world performance if synthetic data mimics real PDF characteristics

---

### 6. **Larger Model Capacity**

**Current:** `ref_feature_dim=512` (~58M parameters)
**Recommended:** `ref_feature_dim=768` (~90M) or `1024` (~130M)

```bash
python train_pattern_segmentation.py \
    --ref-feature-dim 768 \
    --batch-size 32 \  # Adjust based on memory
    --epochs 40
```

**Why it works:**
- More capacity for complex pattern matching
- Better cross-attention representations
- Diminishing returns above 1024

**Expected improvement:** +1-2% IoU → **0.93-0.94 IoU**

**Note:** Larger models need more data/epochs to converge fully

---

### 7. **Enhanced Data Augmentation**

**Current augmentations:**
- Horizontal/vertical flips (50%)
- Random rotation (90°, 180°, 270°)

**Add these:**

```python
self.transform = A.Compose([
    # Existing
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.0),

    # New augmentations
    A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1,
        hue=0.05,
        p=0.5
    ),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.CoarseDropout(
        max_holes=8,
        max_height=32,
        max_width=32,
        p=0.3
    ),
    A.ElasticTransform(
        alpha=50,
        sigma=5,
        p=0.2
    ),
])
```

**Why it works:**
- Better generalization to real PDFs
- Prevents overfitting
- Handles varied image qualities

**Expected improvement:** +1-2% IoU on real data

---

### 8. **Loss Function Tuning**

**Current:** `bce_weight=0.5, dice_weight=0.5`

**Experiment with:**
- `bce_weight=0.3, dice_weight=0.7` - Emphasize overlap (better for IoU)
- `bce_weight=0.4, dice_weight=0.6` - Balanced

**Also try:**
- Focal Loss (handles class imbalance)
- Lovász-Softmax Loss (directly optimizes IoU)
- Tversky Loss (adjustable FP/FN trade-off)

**Expected improvement:** +0.5-1% IoU

---

## Advanced Techniques (More Work)

### 9. **Test-Time Augmentation (TTA)**

Apply augmentations during inference, average predictions:

```python
def predict_with_tta(model, image, reference):
    preds = []

    # Original
    pred = model(image, reference)
    preds.append(pred)

    # Horizontal flip
    pred_hflip = model(
        torch.flip(image, [-1]),
        reference
    )
    preds.append(torch.flip(pred_hflip, [-1]))

    # Vertical flip
    pred_vflip = model(
        torch.flip(image, [-2]),
        reference
    )
    preds.append(torch.flip(pred_vflip, [-2]))

    # Average all predictions
    return torch.stack(preds).mean(dim=0)
```

**Expected improvement:** +2-4% IoU (inference only, no retraining needed!)

---

### 10. **Model Ensembling**

Train 3-5 models with:
- Different random seeds
- Different architectures (vary `ref_feature_dim`)
- Different augmentation strategies

Average their predictions during inference.

**Expected improvement:** +2-3% IoU
**Cost:** 3-5× training time

---

### 11. **Self-Training / Pseudo-Labeling**

1. Train on labeled data
2. Predict on unlabeled real PDFs
3. Keep high-confidence predictions (IoU > 0.9)
4. Add to training set
5. Retrain

**Why it works:**
- Closes domain gap between synthetic and real data
- Increases training data size

**Expected improvement:** +2-4% on real PDFs

---

### 12. **Architecture Improvements**

**Upgrade Reference Encoder:**
- ResNet50 → ResNet101 (more capacity)
- Try EfficientNet-B4/B5 (better efficiency)
- Vision Transformer (ViT) for reference encoding

**Upgrade Decoder:**
- Add more attention heads (4 → 8)
- Deeper decoder blocks
- Feature Pyramid Network (FPN)

**Expected improvement:** +2-3% IoU
**Cost:** Significant code refactoring

---

## Recommended Action Plan

### Phase 1: Quick Experiments (1-2 days)

1. **Batch size sweep:** Try 16, 32, 64, 128
2. **Longer training:** 40 epochs with cosine LR
3. **Loss tuning:** Test different BCE/Dice ratios

**Target:** 0.90 → **0.92-0.93 IoU**

### Phase 1.5: Data Generation (2-3 days)

1. **Generate 10x more synthetic data:** 30,000 images
2. **Target failure cases:** Analyze low-IoU samples, generate similar patterns
3. **Retrain with larger dataset**

**Target:** 0.92-0.93 → **0.93-0.95 IoU**

### Phase 2: Model Scaling (3-4 days)

1. **Larger model:** `ref_feature_dim=768`
2. **Enhanced augmentation**
3. **TTA for evaluation**

**Target:** 0.92-0.93 → **0.94-0.95 IoU**

### Phase 3: Advanced (1 week)

1. **Self-training on real PDFs**
2. **Model ensemble**
3. **Architecture search**

**Target:** 0.94-0.95 → **0.95-0.96+ IoU**

---

## Hardware Utilization

### Current Usage:
- Batch size: 4
- GPU memory: ~10-15GB / 480GB (**3% utilization!**)
- Training time: ~5 min/epoch

### Optimized Usage:
- Batch size: 128
- GPU memory: ~200-250GB / 480GB (50% utilization)
- Training time: ~8-10 min/epoch
- **Better convergence, higher final accuracy**

---

## Expected Final Performance

| Configuration | Val IoU | Real PDF IoU | Effort |
|---------------|---------|--------------|--------|
| **Current** | 0.90 | 0.75 | - |
| Phase 1 (Quick wins) | 0.92-0.93 | 0.78-0.80 | Low |
| Phase 2 (Model scaling) | 0.94-0.95 | 0.82-0.85 | Medium |
| Phase 3 (Advanced) | 0.95-0.96+ | 0.85-0.88+ | High |

---

## Notes

- Real PDF performance gap is partially due to **automatic reference patch selection**
- With human-selected patches, real PDF IoU would be **~5-8% higher**
- Model is already production-ready at 0.90 IoU
- Improvements target: edge cases, difficult patterns, domain adaptation

---

## References

- **Cosine Annealing:** [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
- **Test-Time Augmentation:** Commonly used in Kaggle competitions
- **Self-Training:** [Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method](https://www.researchgate.net/publication/280581078_Pseudo-Label_The_Simple_and_Efficient_Semi-Supervised_Learning_Method_for_Deep_Neural_Networks)
- **Focal Loss:** [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
