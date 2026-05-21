import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, Subset

from src.train import train
from src.eval import evaluate
from mnist.dataloader import get_dataloaders
from mnist.model import SimpleCNN_MNIST

MODEL_NAME = "SimpleCNN_MNIST"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIGIT_A = 4
DIGIT_B = 9

BASE_CONFIG = {
    "data_dir":        "data/raw",
    "batch_size":      64,
    "num_epochs":      10,
    "learning_rate":   0.001,
    "model_class":     SimpleCNN_MNIST,
    "optimizer_class": torch.optim.Adam,
    "criterion":       nn.CrossEntropyLoss(),
    "save_each_epoch": False,
    "digit_a":         DIGIT_A,
    "digit_b":         DIGIT_B,
    "patch_h":         8,
    "patch_w":         8,
    "patch_row":       20,
    "patch_col":       0,
    #"patch2_row": 0,
    #"patch2_col": 20,   
    "patch_color":     (0, 255, 0),
    "n_capas":         3,
    "patience": 3
}

SUBGROUP_LABELS = {
    (0, 0): f"Sin patch, dígito {DIGIT_A}",
    (0, 1): f"Sin patch, dígito {DIGIT_B}",
    (1, 0): f"Con patch, dígito {DIGIT_A}",
    (1, 1): f"Con patch, dígito {DIGIT_B}",
}

P1_VALUES  = [0, 0.01, 0.05]
N_CORRIDAS = 1
SEEDS      = [42]#[42, 123, 999]


def evaluate_subgroups(model, test_ds, test_bal, device, batch_size=64):
    criterion = nn.CrossEntropyLoss()
    results = {}
    for key, indices in test_bal.items():
        loader = DataLoader(Subset(test_ds, indices), batch_size=batch_size)
        _, acc = evaluate(model, loader, criterion, device)
        results[key] = {"acc": acc, "n": len(indices)}
    return results


all_results = []

for p1 in P1_VALUES:
    p2 = round(1 - p1, 10)
    for corrida, seed in enumerate(SEEDS, 1):
        print(f"\np1={p1}, p2={p2}, corrida={corrida}, seed={seed}")

        config = {
            **BASE_CONFIG,
            "p1":   p1,
            "p2":   p2,
            "seed": seed,
        }

        n_capas      = BASE_CONFIG["n_capas"]
        exp_name     = f"{MODEL_NAME}_{DIGIT_A}vs{DIGIT_B}_{n_capas}capas"
        model_dir = f"results/MNIST/{exp_name}/modelos"
        os.makedirs(model_dir, exist_ok=True)
        config["model_save_path"] = f"{model_dir}/p1{p1}_p2{p2}_corrida{corrida}_seed{seed}.pth"

        train_loader, val_loader, test_loader, test_ds, test_bal = get_dataloaders(config)
        model = train(config, train_loader, val_loader, DEVICE)

        test_loss, test_acc = evaluate(model, test_loader, config["criterion"], DEVICE)
        subgroup_results = evaluate_subgroups(model, test_ds, test_bal, DEVICE)

        print(f"  Test acc: {test_acc:.4f}")
        for key, res in subgroup_results.items():
            print(f"  {SUBGROUP_LABELS[key]}: acc={res['acc']:.4f} (n={res['n']})")

        all_results.append({
            "p1": p1, "p2": p2, "corrida": corrida, "seed": seed,
            "test_acc": test_acc,
            "subgroup_results": subgroup_results,
        })

results_dir = f"results/MNIST/{exp_name}/resultados"
os.makedirs(results_dir, exist_ok=True)
results_path = f"{results_dir}/results.pkl"
with open(results_path, "wb") as f:
    pickle.dump(all_results, f)
print(f"\nResultados guardados en {results_path}")
