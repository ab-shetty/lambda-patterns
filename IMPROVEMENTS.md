# Model Improvement Strategies

## Training History

| Run | Dataset | Epochs | Best Val IoU | Best F1 | Time | Notes |
|-----|---------|--------|-------------|---------|------|-------|
| Run 1 | 3K images | 22 | 0.8993 | 0.9337 | ~7.5hr | Baseline, step_size=8 |
| Run 2 | 15K images | 7 (abandoned) | 0.9174 | 0.9448 | - | Too slow, step_size=8 |
| Run 3 | 15K images | 18 | 0.9346 | 0.9565 | ~3.3hr | step_size=5 ✅ |
| **Run 4** | **15K images** | **18** | **0.9344** | **0.9553** | **~3.9hr** | **dim=768 — no improvement** |

**Current best: 0.9346 validation IoU (93.46%)** (Run 3, dim=512)
Real-world performance: **0.82 median IoU** on real PDFs (automatic patch selection)

**Key finding from Runs 3 & 4:** Model plateaus at ~0.934-0.935 regardless of model size. dim=768 matched dim=512 exactly (0.9344 vs 0.9346) while being 20% slower. **The bottleneck is the data, not model capacity.** To push past 0.935, more synthetic data is the only lever.

This document outlines strategies to push performance higher, leveraging the GH200's massive compute capabilities.

---

## Quick Wins (Minimal Code Changes)

### 1. **Batch Size & Data Loading** ✅ (Settled)

- `batch_size=16` — max before OOM on GH200 480GB due to cross-attention at 128×128 resolution
- `num_workers=8` — halved epoch time from 23 min → 11 min
- `image_size=512` — keeping full resolution for quality

This is the settled configuration. No further tuning needed here.

---

### 2. **LR Schedule Tuning** ✅ (Done — confirmed working)

**Was:** `step_size=8, gamma=0.2` (drops at epochs 8, 16)
**Now:** `step_size=5, gamma=0.2` (drops at epochs 4, 9, 14) — use `--step-size 5`

Run 3 confirmed this hits the natural plateau points perfectly with 15K images.

**Still to try:** Cosine Annealing with Warmup for smoother decay. May squeeze +0.5% IoU.

---

### 3. **Epoch Count**

**Finding from Run 3:** Model hard-plateaus at ~0.934 train and val IoU after epoch 11. More epochs yield no gains — the ceiling is the architecture/dataset, not training time.

**Recommendation:** 16-18 epochs with step_size=5 is the sweet spot for 15K images. Don't go longer without changing something else (more data or larger model).

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

### 6. ~~**Larger Model Capacity**~~ — Ruled Out

**Tested:** dim=512 vs dim=768 on 15K images
**Result:** 0.9346 vs 0.9344 — identical, within noise. 20% slower training for zero gain.

**Conclusion:** dim=512 is the settled configuration. The architecture has sufficient capacity for the current dataset. Larger models will not help until the dataset is significantly expanded (30K+).

---

### 7. **Enhanced Data Augmentation**

**Context:** Model is only used on electronic PDFs (never scanned). This rules out scan-artifact augmentations.

**❌ Do NOT add (not present in electronic PDFs):**
- Gaussian noise
- Blur (Gaussian, Motion)
- JPEG compression artifacts
- Elastic transforms
- Perspective/distortion transforms
- Heavy brightness variation
- GridDistortion
- Scale/zoom variations (users always select the full pattern region, so scale is consistent)

**Current augmentations (keep):**
- Horizontal/vertical flips (50%)
- Random rotation (90°, 180°, 270°)

**Add:**

```python
self.transform = A.Compose([
    # Existing
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.0),

    # Color variation — patterns come in many colors
    A.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.2,
        hue=0.1,
        p=0.5
    ),

    # Occlusion — text, images, and other elements realistically cover parts of patterns
    A.CoarseDropout(
        max_holes=8,
        max_height=32,
        max_width=32,
        p=0.3
    ),
])
```

**Why it works:**
- Color jitter handles the variety of pattern colors in real documents
- CoarseDropout simulates other page elements (text blocks, images) occluding patterns
- Nothing here introduces artifacts that don't exist in real electronic PDFs

**Expected improvement:** +1-2% real PDF IoU (better domain match)

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

### Phase 1: Quick Experiments ✅ DONE

1. ~~Batch size sweep~~ — capped at 16 (OOM from cross-attention)
2. ✅ `step_size=5`, 18 epochs → **0.9346 val IoU**
3. ✅ `num_workers=8` → 2x faster training
4. Loss tuning (BCE/Dice ratio) — **not yet tried**

**Achieved:** 0.90 → **0.9346 IoU** ✅

### Phase 1.5: Data Generation (next step)

1. **Generate 30,000 synthetic images** (2x current)
2. **Target failure cases:** Use low-IoU visualizations to identify gaps
3. **Retrain:** `--batch-size 16 --epochs 18 --step-size 5`

**Target:** 0.9346 → **0.95+ IoU**
**Why:** Model has plateaued at current data ceiling — more data is the clearest path forward.

### Phase 2: Augmentation + More Data

1. **Add color jitter + CoarseDropout** — may improve real PDF IoU even if val stays flat
2. **Generate 30K+ synthetic images** — only thing that will move val IoU past 0.935
3. ~~Larger model (dim=768)~~ — ruled out, no benefit on 15K dataset
4. ~~TTA~~ — ruled out, negligible benefit

**Target:** 0.9346 → **0.95+ val IoU**, **0.85+ real PDF IoU**

### Phase 3: Advanced (1 week)

1. **Self-training on real PDFs**
2. **Model ensemble**
3. **Architecture improvements** (ResNet101, more attention heads) — only worth trying after 30K+ dataset

**Target:** 0.96-0.97 → **0.97+ IoU**

---

## Hardware Utilization

### Current Usage (Run 3):
- Batch size: 16 (max before OOM — constrained by cross-attention)
- num_workers: 8
- GPU memory: ~30-40GB / 480GB (~8% utilization)
- Training time: **~11 min/epoch** (~3.3hr for 18 epochs)

### Why OOM at batch_size=16 Despite 480GB:
Cross-attention in dec2 at 128×128 resolution generates attention matrices ~3GB per forward pass, plus gradients. Memory scales with batch size faster than expected.

### Settled Configuration:
- `batch_size=16`, `image_size=512`, `num_workers=8`
- Focus improvements on data and architecture, not hardware tuning

---

## Expected Final Performance

| Configuration | Val IoU | Real PDF IoU | Effort |
|---------------|---------|--------------|--------|
| Run 1 (3K images) | 0.8993 | ~0.75 | - |
| **Run 3 (15K images)** | **0.9346** | **~0.80?** | - |
| Phase 1.5 (augmentation only) | ~0.935 | 0.83-0.85 | Low |
| Phase 2 (30K images + augmentation) | 0.95+ | 0.85-0.87 | Medium |
| Phase 3 (self-training + ensemble) | 0.97+ | 0.89+ | High |

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
