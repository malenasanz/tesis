import torch
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from src.datasets import CelebADataSet
from src.dataloader import split_balanceado
from src.model import SimpleCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FPR_GRID = np.linspace(0, 1, 100)

BIAS_ATTR   = "Young"
TARGET_ATTR = "Blond_Hair"
PKL_PATH    = f"results/results_{BIAS_ATTR}_cruzado.pkl"

BIAS_LABELS = {0: f"no {BIAS_ATTR}", 1: BIAS_ATTR}
BIAS_KEYS   = [0, 1]

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

P1_VALUES    = data["p1_values"]
seeds_usadas = data["seeds"]

# Cargar dataset una sola vez
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
dataset = CelebADataSet(
    img_dir="data/raw/celeba/img_align_celeba",
    attr_path="data/raw/celeba/list_attr_celeba.txt",
    transform=transform,
    target_attr=TARGET_ATTR
)
name_to_idx = {name: i for i, name in enumerate(dataset.images)}


def get_test_data(seed):
    _, _, test_bal = split_balanceado(dataset.df, BIAS_ATTR, TARGET_ATTR, seed)
    # test_bal order: (0,0), (0,1), (1,0), (1,1) → bias values: 0,0,1,1
    bias_values = [0, 0, 1, 1]
    all_test, bias_per_sample = [], []
    for bias_val, idx_arr in zip(bias_values, test_bal):
        all_test.extend(idx_arr)
        bias_per_sample.extend([bias_val] * len(idx_arr))
    test_dataset = Subset(dataset, [name_to_idx[n] for n in all_test])
    loader = DataLoader(test_dataset, batch_size=64)
    return loader, bias_per_sample


def get_probas(model, loader):
    model.eval()
    all_probas, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images).cpu().numpy().flatten()
            all_probas.extend(outputs)
            all_labels.extend(labels.numpy())
    return np.array(all_probas), np.array(all_labels)


# ── Gráfico 1: ROC global por p1 ──────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 7))

# ── Gráfico 2: ROC por subgrupo, un subplot por p1 ───────────────────────────
n_cols = 4
n_rows = math.ceil(len(P1_VALUES) / n_cols)
#fig2, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
#axes = axes.flatten()

for idx_p1, p1 in enumerate(P1_VALUES):
    p2 = round(1 - p1, 1)
    tprs_global = []
    tprs_bias = {b: [] for b in BIAS_KEYS}

    for i, seed in enumerate(seeds_usadas[p1]):
        model_path = f"models/{BIAS_ATTR}_p1{p1}_p2{p2}_corrida{i+1}_seed{seed}.pth"

        model = SimpleCNN().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        loader, bias_per_sample = get_test_data(seed)
        probas, labels = get_probas(model, loader)

        # ROC global
        fpr, tpr, _ = roc_curve(labels, probas)
        tprs_global.append(np.interp(FPR_GRID, fpr, tpr))

        # ROC por grupo de bias
        for b in BIAS_KEYS:
            mask = np.array([s == b for s in bias_per_sample])
            if mask.sum() > 0 and len(np.unique(labels[mask])) > 1:
                fpr_s, tpr_s, _ = roc_curve(labels[mask], probas[mask])
                tprs_bias[b].append(np.interp(FPR_GRID, fpr_s, tpr_s))

    # Grafico 1 - curva global
    mean_tpr = np.mean(tprs_global, axis=0)
    std_tpr  = np.std(tprs_global, axis=0)
    mean_auc = auc(FPR_GRID, mean_tpr)
    ax1.plot(FPR_GRID, mean_tpr, label=f"p1={p1:.1f} (AUC={mean_auc:.3f})")
    ax1.fill_between(FPR_GRID, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.1)

    # Grafico 2 - subplot por p1 con 2 curvas (bias=0 y bias=1)
    '''
    ax2 = axes[idx_p1]
    for b in BIAS_KEYS:
        if tprs_bias[b]:
            mean_tpr_s = np.mean(tprs_bias[b], axis=0)
            mean_auc_s = auc(FPR_GRID, mean_tpr_s)
            ax2.plot(FPR_GRID, mean_tpr_s, label=f"{BIAS_LABELS[b]} ({mean_auc_s:.2f})")
    ax2.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax2.set_title(f"p1={p1:.1f}, p2={p2:.1f}")
    ax2.set_xlabel("FPR")
    ax2.set_ylabel("TPR")
    ax2.legend(fontsize=6)
    ax2.grid(True)

# Ocultar subplots vacíos
for j in range(len(P1_VALUES), len(axes)):
    axes[j].set_visible(False)
'''

ax1.plot([0, 1], [0, 1], "k--", label="Random")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title(f"ROC global vs p1 (bias={BIAS_ATTR}, target={TARGET_ATTR})")
ax1.legend(loc="lower right", fontsize=8)
ax1.grid(True)

fig1.tight_layout()
fig1.savefig(f"results/roc_global_{BIAS_ATTR}.png", dpi=150)

'''
fig2.suptitle(f"ROC por subgrupo (bias={BIAS_ATTR}, target={TARGET_ATTR})", fontsize=14)
fig2.tight_layout()
fig2.savefig(f"results/roc_subgrupos_{BIAS_ATTR}.png", dpi=150)

plt.show()
print(f"Gráficos guardados en results/")
'''

