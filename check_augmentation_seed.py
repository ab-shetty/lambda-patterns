#!/usr/bin/env python3
"""
Check whether seed-resetting produces identical augmentation output
for image and reference patch — tested separately for spatial and color transforms.
"""

import random
import numpy as np
import albumentations as A


def check(name, transform, image):
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    out1 = transform(image=image)['image']

    np.random.seed(seed)
    random.seed(seed)
    out2 = transform(image=image)['image']

    if np.array_equal(out1, out2):
        print(f"PASS [{name}]: identical output with same seed")
    else:
        diff = np.abs(out1.astype(int) - out2.astype(int))
        print(f"FAIL [{name}]: outputs differ — max pixel diff: {diff.max()}, mean diff: {diff.mean():.4f}")


# Asymmetric image so flips/rotations produce detectable differences
image = np.zeros((224, 224, 3), dtype=np.uint8)
image[:112, :112] = [200, 100, 50]
image[:112, 112:] = [50, 200, 100]
image[112:, :112] = [100, 50, 200]
image[112:, 112:] = [150, 150, 50]

check("spatial only", A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.0),
]), image)

check("color only", A.Compose([
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.2, hue=0.1, p=1.0),
]), image)

check("spatial + color", A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.0),
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.2, hue=0.1, p=1.0),
]), image)
