#!/usr/bin/env python3
"""
Check whether seed-resetting produces identical ColorJitter output
for image and reference patch.
"""

import random
import numpy as np
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.0),
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.2, hue=0.1, p=1.0),
])

# Use a fixed test image (solid color so any shift is visible)
image = np.full((224, 224, 3), 128, dtype=np.uint8)
# Add a gradient so rotation/flip differences are detectable
image[:, :112] = 100
image[112:] = 160

seed = 42

np.random.seed(seed)
random.seed(seed)
out1 = transform(image=image)['image']

np.random.seed(seed)
random.seed(seed)
out2 = transform(image=image)['image']

if np.array_equal(out1, out2):
    print("PASS: identical output with same seed")
else:
    diff = np.abs(out1.astype(int) - out2.astype(int))
    print(f"FAIL: outputs differ — max pixel diff: {diff.max()}, mean diff: {diff.mean():.4f}")
    print(f"  out1 mean: {out1.mean():.2f}, out2 mean: {out2.mean():.2f}")
