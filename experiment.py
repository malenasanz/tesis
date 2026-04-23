import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.dataloader import get_dataloaders
from src.train import train
from src.eval import evaluate_subgroups
from src.model import SimpleCNN

BASE_CONFIG = {
    "img_dir": "data/raw/celeba/img_align_celeba",
    "attr_path": "data/raw/celeba/list_attr_celeba.txt",
    "target_attr": "Blond_Hair",
    "bias_attr": "Male",
    "p2": 0.5,
    "batch_size": 64,
    "num_epochs": 4,
    "learning_rate": 0.001,
    "seed": 42,
    "model_class": SimpleCNN,
    "optimizer_class": torch.optim.Adam,
    "criterion": nn.BCELoss(),
}

SUBGROUP_LABELS = {
    (0, 0): "Mujer no rubia",
    (0, 1): "Mujer rubia",
    (1, 0): "Hombre no rubio",
    (1, 1): "Hombre rubio",
}

P1_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# results[p1][(bias, target)] = acc
results = {}

for p1 in P1_VALUES:
    config = {**BASE_CONFIG, "p1": p1}
    print(f"\n--- p1={p1} | p2={config['p2']} ---")

    train_loader, val_loader, test_loader, dataset, test_df = get_dataloaders(config)
    model = train(config, train_loader, val_loader, device)

    subgroup_results = evaluate_subgroups(model, dataset, test_df, config["bias_attr"], config["target_attr"], device)
    results[p1] = {k: v["acc"] for k, v in subgroup_results.items()}
    print("  Test subgroups:", {SUBGROUP_LABELS[k]: f"{v:.4f}" for k, v in results[p1].items()})

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for key, label in SUBGROUP_LABELS.items():
    accs = [results[p1][key] for p1 in P1_VALUES]
    ax.plot(P1_VALUES, accs, marker="o", label=label)

ax.set_xlabel("p1 (proporción de hombres dentro de rubios)")
ax.set_ylabel("Accuracy en test")
ax.set_title(f"Accuracy por subgrupo vs p1 (p2={BASE_CONFIG['p2']})")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("results/accuracy_vs_p1.png", dpi=150)
plt.show()
print("Gráfico guardado en results/accuracy_vs_p1.png")
