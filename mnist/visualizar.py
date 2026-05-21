import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mnist.datasets import MNISTBinary

DIGIT_A      = 4
DIGIT_B      = 9
SEED         = 42
N_COLS       = 6
PATCH_H      = 8
PATCH_W      = 8
PATCH_ROW    = 0
PATCH_COL    = 0
PATCH_COLOR  = (255, 255, 255)
NOISE_STD    = 0
MAX_ROTATION = 0

rng = np.random.default_rng(SEED)

test_ds = MNISTBinary(root="data/raw", train=False, transform=None,
                      digit_a=DIGIT_A, digit_b=DIGIT_B,
                      noise_std=NOISE_STD, max_rotation=MAX_ROTATION, seed=SEED,
                      patch_h=PATCH_H, patch_w=PATCH_W, patch_row=PATCH_ROW,
                      patch_col=PATCH_COL, patch_color=PATCH_COLOR)

idx_0 = np.where(test_ds.labels == 0)[0]
idx_1 = np.where(test_ds.labels == 1)[0]
rng.shuffle(idx_0); rng.shuffle(idx_1)
n_per_group = min(len(idx_0), len(idx_1)) // 2
idx_0 = idx_0[:n_per_group * 2]
idx_1 = idx_1[:n_per_group * 2]

patched_0 = rng.choice(idx_0, size=n_per_group, replace=False)
patched_1 = rng.choice(idx_1, size=n_per_group, replace=False)
patch_indices = np.concatenate([patched_0, patched_1])
test_ds.set_patches(patch_indices)

subgroups = {
    (0, 0): idx_0[~np.isin(idx_0, patch_indices)],
    (0, 1): idx_1[~np.isin(idx_1, patch_indices)],
    (1, 0): idx_0[ np.isin(idx_0, patch_indices)],
    (1, 1): idx_1[ np.isin(idx_1, patch_indices)],
}

labels = {
    (0, 0): f"Sin fondo verde, dígito {DIGIT_A}",
    (0, 1): f"Sin fondo verde, dígito {DIGIT_B}",
    (1, 0): f"Con fondo verde, dígito {DIGIT_A}",
    (1, 1): f"Con fondo verde, dígito {DIGIT_B}",
}

fig, axes = plt.subplots(4, N_COLS, figsize=(N_COLS * 2, 9))
for row, (key, title) in enumerate(labels.items()):
    indices = subgroups[key]
    sample  = rng.choice(indices, size=N_COLS, replace=False)
    axes[row, 0].set_ylabel(title, fontsize=10)
    for col, idx in enumerate(sample):
        img, _ = test_ds[idx]
        axes[row, col].imshow(img)
        axes[row, col].axis("off")

plt.suptitle(f"Muestras por subgrupo — dígitos {DIGIT_A} vs {DIGIT_B}", fontsize=13)
plt.tight_layout()
plt.show()
