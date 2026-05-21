import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.model import SimpleCNN
from src.datasets import CelebADataSet
from src.dataloader import split_balanceado

# ── Configuración ──────────────────────────────────────────────────────────────
BIAS_ATTR   = "Male"
TARGET_ATTR = "Blond_Hair"
P1          = 0.5
P2          = 0.5
SEED        = 6550
MODEL_NAME  = "SimpleCNN_3capas"
MODEL_PATH  = f"results/{BIAS_ATTR}/{MODEL_NAME}/modelos/p1{P1}_p2{P2}_corrida1_seed{SEED}.pth"
N_IMGS      = 4
IMG_OFFSET  = 12  # cambiá a 4, 8, 12... para ver otras imágenes
SHOW_TARGET = None  # 0 = solo no rubios, 1 = solo rubios, None = todos

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

# ── Cargar dataset y test set ──────────────────────────────────────────────────
transform     = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
transform_pil = transforms.Resize((128, 128))

dataset = CelebADataSet(
    img_dir="data/raw/celeba/img_align_celeba",
    attr_path="data/raw/celeba/list_attr_celeba.txt",
    transform=transform,
    target_attr=TARGET_ATTR
)
name_to_idx = {name: i for i, name in enumerate(dataset.images)}
_, _, test_bal = split_balanceado(dataset.df, BIAS_ATTR, TARGET_ATTR, SEED)

all_subgroups = list(zip(SUBGROUP_KEYS, test_bal))
subgroups_show = [(k, arr) for k, arr in all_subgroups if SHOW_TARGET is None or k[1] == SHOW_TARGET]

# ── GradCAM — tres capas ───────────────────────────────────────────────────────
N_COLS = 4  # Original | conv1 | conv2 | conv3
fig, axes = plt.subplots(len(subgroups_show), N_IMGS * N_COLS,
                         figsize=(N_IMGS * N_COLS * 2.5, len(subgroups_show) * 3))

with GradCAM(model=model, target_layers=[model.conv1]) as cam1, \
     GradCAM(model=model, target_layers=[model.conv2]) as cam2, \
     GradCAM(model=model, target_layers=[model.conv3]) as cam3:

    for row, (key, idx_arr) in enumerate(subgroups_show):
        for col, img_name in enumerate(idx_arr[IMG_OFFSET:IMG_OFFSET+N_IMGS]):
            img_path = f"data/raw/celeba/img_align_celeba/{img_name}"
            pil_img  = transform_pil(Image.open(img_path).convert("RGB"))
            rgb_img  = np.array(pil_img).astype(np.float32) / 255.0
            input_tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logit = model(input_tensor).item()
            prob = torch.sigmoid(torch.tensor(logit)).item()
            pred = 1 if logit > 0 else 0
            true_label = key[1]  # target_attr es la segunda coordenada
            correcto = "✓" if pred == true_label else "✗"
            print(f"  {SUBGROUP_LABELS[key]} | img {col+1}: P(rubio)={prob:.4f} {correcto}")

            targets = [BinaryClassifierOutputTarget(true_label)]
            vis1 = show_cam_on_image(rgb_img, cam1(input_tensor=input_tensor, targets=targets)[0], use_rgb=True)
            vis2 = show_cam_on_image(rgb_img, cam2(input_tensor=input_tensor, targets=targets)[0], use_rgb=True)
            vis3 = show_cam_on_image(rgb_img, cam3(input_tensor=input_tensor, targets=targets)[0], use_rgb=True)

            base = col * N_COLS
            for ax_idx, (img_data, title) in enumerate([(rgb_img, "Original"), (vis1, "conv1"), (vis2, "conv2"), (vis3, "conv3")]):
                ax = axes[row, base + ax_idx]
                ax.imshow(img_data)
                ax.axis("off")
                if ax_idx == 0:
                    color = "green" if pred == true_label else "red"
                    label_str = "rubio" if true_label == 1 else "no rubio"
                    ax.text(0.5, -0.05, f"Real: {label_str} | P={prob:.2f} {correcto}",
                            ha="center", va="top", transform=ax.transAxes, fontsize=7, color=color)
                if row == 0 and col == 0:
                    ax.set_title(title, fontsize=8)
            if col == 0:
                axes[row, base].set_ylabel(SUBGROUP_LABELS[key], fontsize=9)

fig.suptitle(f"GradCAM por capa — bias={BIAS_ATTR}, target={TARGET_ATTR}", fontsize=12)
plt.tight_layout()
out_path = f"results/{BIAS_ATTR}/{MODEL_NAME}/gradcam/p1{P1}_p2{P2}_seed{SEED}/offset{IMG_OFFSET}.png"
os.makedirs(f"results/{BIAS_ATTR}/{MODEL_NAME}/gradcam/p1{P1}_p2{P2}_seed{SEED}", exist_ok=True)
plt.savefig(out_path, dpi=150)
plt.show()
print(f"Guardado en {out_path}")


# ── Comparación de capas (comentado) ──────────────────────────────────────────
# N_COLS = 4  # Original | conv1 | conv2 | conv3
# with GradCAM(model=model, target_layers=[model.conv1]) as cam1, \
#      GradCAM(model=model, target_layers=[model.conv2]) as cam2, \
#      GradCAM(model=model, target_layers=[model.conv3]) as cam3:
#     ...


# ── GradCAM por epoch (comentado) ─────────────────────────────────────────────
# def graficar_gradcam_por_epoch(num_epochs):
#     fig2, axes2 = plt.subplots(len(SUBGROUP_KEYS), num_epochs + 1, ...)
#     for epoch in range(1, num_epochs + 1):
#         epoch_path = MODEL_PATH.replace(".pth", f"_epoch{epoch}.pth")
#         ...
# NUM_EPOCHS = 10
# graficar_gradcam_por_epoch(NUM_EPOCHS)
