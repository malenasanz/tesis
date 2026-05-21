import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from mnist.datasets import MNISTBinary
from mnist.model import SimpleCNN_MNIST

# ── Configuración ──────────────────────────────────────────────────────────────
DIGIT_A    = 4
DIGIT_B    = 9
MODEL_NAME = "SimpleCNN_MNIST"
P1         = 0
P2         = 1 - P1
CORRIDA    = 1
SEED       = 42
N_IMGS       = 4
PATCH_H      = 8
PATCH_W      = 8
PATCH_ROW    = 20
PATCH_COL    = 0
#patch_row_2 = 0 
#patch_col_2 = 20
PATCH_COLOR  = (0, 255, 0)
NOISE_STD    = 0
N_CAPAS      = 3

MODEL_PATH = f"results/MNIST/{MODEL_NAME}_{DIGIT_A}vs{DIGIT_B}_{N_CAPAS}capas/modelos/p1{P1}_p2{P2}_corrida{CORRIDA}_seed{SEED}.pth"
BASE_DIR   = f"results/MNIST/{MODEL_NAME}_{DIGIT_A}vs{DIGIT_B}_{N_CAPAS}capas"

SUBGROUP_LABELS = {
    (0, 0): f"Sin patch, dígito {DIGIT_A}",
    (0, 1): f"Sin patch, dígito {DIGIT_B}",
    (1, 0): f"Con patch, dígito {DIGIT_A}",
    (1, 1): f"Con patch, dígito {DIGIT_B}",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TowardBTarget:
    """Regiones que aumentan logit_9 relativo a logit_4."""
    def __call__(self, model_output):
        return model_output[1] - model_output[0]


class TowardATarget:
    """Regiones que aumentan logit_4 relativo a logit_9."""
    def __call__(self, model_output):
        return model_output[0] - model_output[1]

# ── Cargar modelo ──────────────────────────────────────────────────────────────
model = SimpleCNN_MNIST(n_capas=N_CAPAS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ── Armar test set balanceado ──────────────────────────────────────────────────
rng     = np.random.default_rng(SEED)
test_ds = MNISTBinary(root="data/raw", train=False, transform=transforms.ToTensor(),
                      digit_a=DIGIT_A, digit_b=DIGIT_B, noise_std=NOISE_STD,
                      patch_h=PATCH_H, patch_w=PATCH_W, patch_row=PATCH_ROW,
                      patch_col=PATCH_COL, patch_color=PATCH_COLOR, seed=SEED)

idx_0 = np.where(test_ds.labels == 0)[0]
idx_1 = np.where(test_ds.labels == 1)[0]
rng.shuffle(idx_0); rng.shuffle(idx_1)
n_per_group = min(len(idx_0), len(idx_1)) // 2
idx_0 = idx_0[:n_per_group * 2]
idx_1 = idx_1[:n_per_group * 2]
patched = np.concatenate([
    rng.choice(idx_0, size=n_per_group, replace=False),
    rng.choice(idx_1, size=n_per_group, replace=False),
])
test_ds.set_patches(patched)

subgroups = {
    (0, 0): idx_0[~np.isin(idx_0, patched)],
    (0, 1): idx_1[~np.isin(idx_1, patched)],
    (1, 0): idx_0[ np.isin(idx_0, patched)],
    (1, 1): idx_1[ np.isin(idx_1, patched)],
}

layers      = model.conv_layers()
layer_names = [f"conv{i+1}" for i in range(len(layers))]
N_COLS     = 1 + len(layers)


def generar_gradcam(cam_class, output_dir, titulo):
    cam_ctxs = [cam_class(model=model, target_layers=[l]) for l in layers]
    # N_COLS = original + una por capa + diferencial por capa
    n_cols_total = 1 + len(layers) * 2

    fig, axes = plt.subplots(4, N_IMGS * n_cols_total,
                             figsize=(N_IMGS * n_cols_total * 2, 4 * 2.8))

    for row, (key, label) in enumerate(SUBGROUP_LABELS.items()):
        indices = subgroups[key]
        sample  = rng.choice(indices, size=N_IMGS, replace=False)

        for col, idx in enumerate(sample):
            img_tensor, true_label = test_ds[idx]
            input_tensor = img_tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(input_tensor).squeeze(0)
            probs     = torch.softmax(logits, dim=0)
            pred      = logits.argmax().item()
            prob_pred = probs[pred].item()
            correcto  = "✓" if pred == true_label else "✗"

            rgb = img_tensor.permute(1, 2, 0).numpy().astype(np.float32)

            # GradCAM hacia DIGIT_B y contrastivo firmado
            maps_b    = [cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(1)])[0]
                         for cam in cam_ctxs]
            # mapa firmado: pos = empuja hacia 9, neg = empuja hacia 4
            maps_pos  = [cam(input_tensor=input_tensor, targets=[TowardBTarget()])[0]
                         for cam in cam_ctxs]
            maps_neg  = [cam(input_tensor=input_tensor, targets=[TowardATarget()])[0]
                         for cam in cam_ctxs]
            diffs     = [p - n for p, n in zip(maps_pos, maps_neg)]

            viss = [show_cam_on_image(rgb, m, use_rgb=True) for m in maps_b]

            base       = col * n_cols_total
            color      = "green" if pred == true_label else "red"
            digit_name = DIGIT_A if true_label == 0 else DIGIT_B
            pred_name  = DIGIT_A if pred == 0 else DIGIT_B

            # columna 0: original
            ax = axes[row, base]
            ax.imshow(rgb)
            ax.axis("off")
            ax.set_title(f"Real:{digit_name} Pred:{pred_name}({prob_pred:.2f}){correcto}",
                         fontsize=7, color=color)
            if col == 0:
                ax.set_ylabel(label, fontsize=8)

            # columnas por capa: GradCAM hacia DIGIT_B + diferencial
            for i, (vis, diff, lname) in enumerate(zip(viss, diffs, layer_names)):
                ax_vis = axes[row, base + 1 + i * 2]
                ax_vis.imshow(vis)
                ax_vis.axis("off")
                if row == 0 and col == 0:
                    ax_vis.set_title(f"{lname} →{DIGIT_B}", fontsize=7)

                ax_diff = axes[row, base + 2 + i * 2]
                ax_diff.imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1)
                ax_diff.axis("off")
                if row == 0 and col == 0:
                    ax_diff.set_title(f"{lname} diff\nrojo={DIGIT_B} azul={DIGIT_A}", fontsize=7)

    for cam in cam_ctxs:
        cam.__exit__(None, None, None)

    fig.suptitle(f"{titulo} — {DIGIT_A} vs {DIGIT_B} | p1={P1}, p2={P2}, seed={SEED}", fontsize=11)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/gradcam_p1{P1}_p2{P2}_seed{SEED}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Guardado en {out_path}")


generar_gradcam(GradCAM,       f"{BASE_DIR}/gradcam",    "GradCAM")
generar_gradcam(GradCAMPlusPlus, f"{BASE_DIR}/gradcam++", "GradCAM++")
plt.show()
