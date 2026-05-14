import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.model import SimpleCNN
from src.datasets import CelebADataSet
from src.dataloader import split_balanceado

# ── Configuración ──────────────────────────────────────────────────────────────
BIAS_ATTR   = "Male"
TARGET_ATTR = "Blond_Hair"
P1          = 0.5
P2          = 0.5
SEED        = 4539
MODEL_NAME = "SimpleCNN_3capas"
MODEL_PATH  = f"results/{BIAS_ATTR}/{MODEL_NAME}/modelos/p1{P1}_p2{P2}_corrida1_seed{SEED}.pth"
N_IMGS      = 2

SUBGROUP_LABELS = {
    (0, 0): "Mujer no rubia",
    (0, 1): "Mujer rubia",
    (1, 0): "Hombre no rubio",
    (1, 1): "Hombre rubio",
}
SUBGROUP_KEYS = [(0, 0), (0, 1), (1, 0), (1, 1)]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Cargar modelo ──────────────────────────────────────────────────────────────
model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

target_layers = [model.conv2]

# ── Cargar dataset y test set ──────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
transform_pil = transforms.Resize((128, 128))

dataset = CelebADataSet(
    img_dir="data/raw/celeba/img_align_celeba",
    attr_path="data/raw/celeba/list_attr_celeba.txt",
    transform=transform,
    target_attr=TARGET_ATTR
)
name_to_idx = {name: i for i, name in enumerate(dataset.images)}

_, _, test_bal = split_balanceado(dataset.df, BIAS_ATTR, TARGET_ATTR, SEED)

# ── GradCAM ────────────────────────────────────────────────────────────────────
N_COLS = 4  # Original | GradCAM conv1 | GradCAM conv2 | GradCAM conv3
fig, axes = plt.subplots(len(SUBGROUP_KEYS), N_IMGS * N_COLS, figsize=(N_IMGS * N_COLS * 2.5, len(SUBGROUP_KEYS) * 3))

targets = [ClassifierOutputTarget(0)]

with GradCAM(model=model, target_layers=[model.conv1]) as cam1, \
     GradCAM(model=model, target_layers=[model.conv2]) as cam2, \
     GradCAM(model=model, target_layers=[model.conv3]) as cam3:

    for row, (key, idx_arr) in enumerate(zip(SUBGROUP_KEYS, test_bal)):
        sample_names = idx_arr[:N_IMGS]
        for col, img_name in enumerate(sample_names):
            img_path = f"data/raw/celeba/img_align_celeba/{img_name}"
            pil_img  = transform_pil(Image.open(img_path).convert("RGB"))
            rgb_img  = np.array(pil_img).astype(np.float32) / 255.0
            input_tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                prob = model(input_tensor).item()
            print(f"  {SUBGROUP_LABELS[key]} | img {col+1}: P(rubio)={prob:.4f}")

            vis_conv1 = show_cam_on_image(rgb_img, cam1(input_tensor=input_tensor, targets=targets)[0], use_rgb=True)
            vis_conv2 = show_cam_on_image(rgb_img, cam2(input_tensor=input_tensor, targets=targets)[0], use_rgb=True)
            vis_conv3 = show_cam_on_image(rgb_img, cam3(input_tensor=input_tensor, targets=targets)[0], use_rgb=True)

            base = col * N_COLS
            for ax_idx, (img_data, title) in enumerate([(rgb_img, "Original"), (vis_conv1, "conv1"), (vis_conv2, "conv2"), (vis_conv3, "conv3")]):
                ax = axes[row, base + ax_idx]
                ax.imshow(img_data)
                ax.axis("off")
                ax.text(0.5, -0.05, f"P(rubio)={prob:.2f}" if ax_idx == 0 else "",
                        ha="center", va="top", transform=ax.transAxes, fontsize=8)
                if row == 0:
                    ax.set_title(title, fontsize=8)
            if col == 0:
                axes[row, base].set_ylabel(SUBGROUP_LABELS[key], fontsize=9)

fig.suptitle(f"GradCAM — bias={BIAS_ATTR}, target={TARGET_ATTR}", fontsize=12)
plt.tight_layout()
import os
out_path = f"results/{BIAS_ATTR}/{MODEL_NAME}/gradcam/gradcam_p1{P1}_p2{P2}_seed{SEED}.png"
os.makedirs(f"results/{BIAS_ATTR}/{MODEL_NAME}/gradcam", exist_ok=True)
plt.savefig(out_path, dpi=150)
plt.show()
print(f"Guardado en {out_path}")


def graficar_gradcam_por_epoch(num_epochs):
    """GradCAM de conv3 para cada epoch. Filas = subgrupos, columnas = epochs."""
    fig2, axes2 = plt.subplots(len(SUBGROUP_KEYS), num_epochs + 1,
                               figsize=((num_epochs + 1) * 2.5, len(SUBGROUP_KEYS) * 3))

    targets = [ClassifierOutputTarget(0)]

    for row, (key, idx_arr) in enumerate(zip(SUBGROUP_KEYS, test_bal)):
        img_name = idx_arr[0]
        img_path = f"data/raw/celeba/img_align_celeba/{img_name}"
        pil_img  = transform_pil(Image.open(img_path).convert("RGB"))
        rgb_img  = np.array(pil_img).astype(np.float32) / 255.0
        input_tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)

        # Original
        axes2[row, 0].imshow(rgb_img)
        axes2[row, 0].axis("off")
        axes2[row, 0].set_ylabel(SUBGROUP_LABELS[key], fontsize=9)
        if row == 0:
            axes2[row, 0].set_title("Original", fontsize=8)

        for epoch in range(1, num_epochs + 1):
            epoch_path = MODEL_PATH.replace(".pth", f"_epoch{epoch}.pth")
            epoch_model = SimpleCNN().to(DEVICE)
            epoch_model.load_state_dict(torch.load(epoch_path, map_location=DEVICE))
            epoch_model.eval()

            with torch.no_grad():
                prob = epoch_model(input_tensor).item()

            with GradCAM(model=epoch_model, target_layers=[epoch_model.conv3]) as cam:
                vis = show_cam_on_image(rgb_img, cam(input_tensor=input_tensor, targets=targets)[0], use_rgb=True)

            ax = axes2[row, epoch]
            ax.imshow(vis)
            ax.axis("off")
            ax.text(0.5, -0.05, f"P={prob:.2f}", ha="center", va="top",
                    transform=ax.transAxes, fontsize=8)
            if row == 0:
                ax.set_title(f"Epoch {epoch}", fontsize=8)

    fig2.suptitle(f"GradCAM conv3 por epoch — {SUBGROUP_LABELS}", fontsize=10)
    plt.tight_layout()
    out2 = f"results/{BIAS_ATTR}/{MODEL_NAME}/gradcam/gradcam_epochs_p1{P1}_p2{P2}_seed{SEED}.png"
    plt.savefig(out2, dpi=150)
    plt.show()
    print(f"Guardado en {out2}")


NUM_EPOCHS = 10  # cambiá según cuántas epochs guardaste
graficar_gradcam_por_epoch(NUM_EPOCHS)
