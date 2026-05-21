import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from .datasets import MNISTBinary


def get_dataloaders(config):
    transform = transforms.ToTensor()
    rng = np.random.default_rng(config["seed"])
    p1  = config["p1"]   # proporción de dígito-0 con patch en train
    p2  = config["p2"]   # proporción de dígito-1 con patch en train

    digit_a      = config.get("digit_a", 0)
    digit_b      = config.get("digit_b", 1)
    patch_h      = config.get("patch_h", 4)
    patch_w      = config.get("patch_w", 4)
    noise_std    = config.get("noise_std", 0.0)
    max_rotation = config.get("max_rotation", 0)
    seed         = config.get("seed", 0)
    patch_row    = config.get("patch_row", 0)
    patch_col    = config.get("patch_col", 0)
    patch_color  = config.get("patch_color", (255, 255, 255))
    bg_color    = config.get("bg_color", None)
    patch2_row  = config.get("patch2_row", None)
    patch2_col  = config.get("patch2_col", None)
    patch2_color = config.get("patch2_color", None)
    ds_kwargs = dict(digit_a=digit_a, digit_b=digit_b, patch_h=patch_h, patch_w=patch_w,
                     noise_std=noise_std, max_rotation=max_rotation, seed=seed,
                     patch_row=patch_row, patch_col=patch_col, patch_color=patch_color,
                     patch2_row=patch2_row, patch2_col=patch2_col, patch2_color=patch2_color,
                     bg_color=bg_color)
    train_ds = MNISTBinary(root=config["data_dir"], train=True,  transform=transform, **ds_kwargs)
    test_ds  = MNISTBinary(root=config["data_dir"], train=False, transform=transform, **ds_kwargs)

    # ── Train / val split (80/20, clases balanceadas) ─────────────────────────
    idx_0 = np.where(train_ds.labels == 0)[0]
    idx_1 = np.where(train_ds.labels == 1)[0]
    rng.shuffle(idx_0); rng.shuffle(idx_1)
    n_cls = min(len(idx_0), len(idx_1))  # balance 50/50
    idx_0 = idx_0[:n_cls]
    idx_1 = idx_1[:n_cls]

    n_train = int(n_cls * 0.8)
    train_0, val_0 = idx_0[:n_train], idx_0[n_train:]
    train_1, val_1 = idx_1[:n_train], idx_1[n_train:]

    # ── Asignar patch en train (p1/p2) y val (50/50) ──────────────────────────
    patch_indices = np.concatenate([
        rng.choice(train_0, size=int(len(train_0) * p1), replace=False),
        rng.choice(train_1, size=int(len(train_1) * p2), replace=False),
        rng.choice(val_0,   size=int(len(val_0) * p1),   replace=False),
        rng.choice(val_1,   size=int(len(val_1) * p2),   replace=False),
    ])
    train_ds.set_patches(patch_indices)

    # ── Verificación de subgrupos en train ────────────────────────────────────
    patch_set = set(patch_indices)
    n_0_con    = sum(1 for i in train_0 if i in patch_set)
    n_0_sin    = len(train_0) - n_0_con
    n_1_con    = sum(1 for i in train_1 if i in patch_set)
    n_1_sin    = len(train_1) - n_1_con
    print(f"  Train subgrupos (p1={p1}, p2={p2}):")
    print(f"    dígito {digit_a} sin patch: {n_0_sin} | con patch: {n_0_con}")
    print(f"    dígito {digit_b} sin patch: {n_1_sin} | con patch: {n_1_con}")

    # ── Test: balanceado, 50/50 patch por clase ────────────────────────────────
    test_0 = np.where(test_ds.labels == 0)[0]
    test_1 = np.where(test_ds.labels == 1)[0]
    rng.shuffle(test_0); rng.shuffle(test_1)
    n_per_group = min(len(test_0), len(test_1)) // 2  # exactamente 1/4 del test por subgrupo
    test_0 = test_0[:n_per_group * 2]
    test_1 = test_1[:n_per_group * 2]

    patch_test = np.concatenate([
        rng.choice(test_0, size=n_per_group, replace=False),
        rng.choice(test_1, size=n_per_group, replace=False),
    ])
    test_ds.set_patches(patch_test)

    # Subgrupos de test: (patch, dígito) → índices en test_ds (exactamente n_per_group cada uno)
    test_bal = {
        (0, 0): test_0[~np.isin(test_0, patch_test)],
        (0, 1): test_1[~np.isin(test_1, patch_test)],
        (1, 0): test_0[ np.isin(test_0, patch_test)],
        (1, 1): test_1[ np.isin(test_1, patch_test)],
    }

    bs = config.get("batch_size", 64)
    train_idx = np.concatenate([train_0, train_1])
    val_idx   = np.concatenate([val_0,   val_1])
    test_idx  = np.concatenate([test_0,  test_1])

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=bs, shuffle=True)
    val_loader   = DataLoader(Subset(train_ds, val_idx),   batch_size=bs)
    test_loader  = DataLoader(Subset(test_ds,  test_idx),  batch_size=bs)

    return train_loader, val_loader, test_loader, test_ds, test_bal
