import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from src.dataloader import get_dataloaders
from src.train import train
from src.eval import evaluate_subgroups
from src.model import SimpleCNN

MODEL_NAME = "SimpleCNN_3capas"

BASE_CONFIG = {
    "img_dir": "data/raw/celeba/img_align_celeba",
    "attr_path": "data/raw/celeba/list_attr_celeba.txt",
    "target_attr": "Blond_Hair",
    "bias_attr": "Male",
    "batch_size": 64,
    "num_epochs": 10,
    "learning_rate": 0.001,
    "model_class": SimpleCNN,
    "optimizer_class": torch.optim.Adam,
    "criterion": nn.BCELoss(),
    "save_each_epoch": True
}

SUBGROUP_LABELS = {
    (0, 0): "Mujer no rubia",
    (0, 1): "Mujer rubia",
    (1, 0): "Hombre no rubio",
    (1, 1): "Hombre rubio",
}

P1_VALUES = [0.5]
N_CORRIDAS = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# results[p1][subgroup_key] = [acc_corrida1, acc_corrida2, ...]
results = {p1: {key: [] for key in SUBGROUP_LABELS} for p1 in P1_VALUES}
seeds_usadas = {p1: [] for p1 in P1_VALUES}

os.makedirs(f"results/{BASE_CONFIG['bias_attr']}/{MODEL_NAME}/modelos", exist_ok=True)

for p1 in P1_VALUES:
    p2 = round(1 - p1, 1)
    print(f"\n{'='*50}")
    print(f"p1={p1} (hombres no rubios) | p2={p2} (hombres rubios)")
    print(f"{'='*50}")

    for i in range(N_CORRIDAS):
        seed = np.random.randint(0, 10000)
        seeds_usadas[p1].append(seed)
        config = {**BASE_CONFIG, "p1": p1, "p2": p2, "seed": seed,
                  "model_save_path": f"results/{BASE_CONFIG['bias_attr']}/{MODEL_NAME}/modelos/p1{p1}_p2{p2}_corrida{i+1}_seed{seed}.pth"}
        print(f"\n  Corrida {i+1}/{N_CORRIDAS} | seed={seed}")

        train_loader, val_loader, test_loader, dataset, test_df = get_dataloaders(config)
        model = train(config, train_loader, val_loader, device)
        subgroup_results = evaluate_subgroups(model, dataset, test_df, config["bias_attr"], config["target_attr"], device)

        for key in SUBGROUP_LABELS:
            results[p1][key].append(subgroup_results[key]["acc"])

    print(f"\n  Resumen p1={p1}:")
    for key, label in SUBGROUP_LABELS.items():
        accs = results[p1][key]
        print(f"    {label}: mean={np.mean(accs):.4f} | std={np.std(accs):.4f}")

bias_attr = BASE_CONFIG["bias_attr"]
out_dir = f"results/{bias_attr}/{MODEL_NAME}"
os.makedirs(f"{out_dir}/modelos", exist_ok=True)
with open(f"{out_dir}/resultados/results.pkl", "wb") as f:
    pickle.dump({
        "results":    results,
        "p1_values":  P1_VALUES,
        "seeds":      seeds_usadas,
        "bias_attr":  bias_attr,
        "target_attr": BASE_CONFIG["target_attr"],
        "n_corridas": N_CORRIDAS,
        "num_epochs": BASE_CONFIG["num_epochs"],
    }, f)
print(f"\nResultados guardados en {out_dir}/results.pkl")
