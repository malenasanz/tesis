import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms
from src.datasets import CelebADataSet
from src.model import SimpleCNN

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FPR_GRID = np.linspace(0, 1, 100)

TARGET_ATTRS = ["Male", "Young"]

with open("results/results_simple.pkl", "rb") as f:
    data = pickle.load(f)

seeds_usadas = data["seeds"]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


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


fig, ax = plt.subplots(figsize=(8, 6))

for target_attr in TARGET_ATTRS:
    dataset = CelebADataSet(
        img_dir="data/raw/celeba/img_align_celeba",
        attr_path="data/raw/celeba/list_attr_celeba.txt",
        transform=transform,
        target_attr=target_attr
    )

    tprs = []
    for i, seed in enumerate(seeds_usadas[target_attr]):
        model_path = f"models/simple_{target_attr}_corrida{i+1}_seed{seed}.pth"

        rng = np.random.default_rng(seed)
        n       = len(dataset)
        indices = rng.permutation(n)
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)
        test_idx = indices[n_train + n_val:]

        from torch.utils.data import Subset
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=64)

        model = SimpleCNN().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        probas, labels = get_probas(model, test_loader)
        fpr, tpr, _ = roc_curve(labels, probas)
        tprs.append(np.interp(FPR_GRID, fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr  = np.std(tprs, axis=0)
    mean_auc = auc(FPR_GRID, mean_tpr)

    ax.plot(FPR_GRID, mean_tpr, label=f"{target_attr} (AUC={mean_auc:.3f})")
    ax.fill_between(FPR_GRID, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.15)

ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC promedio por atributo (sin balanceo)")
ax.legend(loc="lower right")
ax.grid(True)
plt.tight_layout()
plt.savefig("results/roc_simple.png", dpi=150)
plt.show()
print("Gráfico guardado en results/roc_simple.png")
