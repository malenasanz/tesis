import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score
from src.dataloader import get_dataloaders_simple
from src.train import train
from src.model import SimpleCNN

print('EMPEZO A CORRER MAIN_SIMPLE')

BASE_CONFIG = {
    "img_dir": "data/raw/celeba/img_align_celeba",
    "attr_path": "data/raw/celeba/list_attr_celeba.txt",
    "batch_size": 64,
    "num_epochs": 10,
    "learning_rate": 0.001,
    "model_class": SimpleCNN,
    "optimizer_class": torch.optim.Adam,
    "criterion": nn.BCELoss(),
}

TARGET_ATTRS = ["Male", "Young"]
N_CORRIDAS   = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

results      = {t: [] for t in TARGET_ATTRS}
seeds_usadas = {t: [] for t in TARGET_ATTRS}

for i in range(N_CORRIDAS):
    seed = np.random.randint(0, 10000)
    print(f"\n{'='*50}")
    print(f"Corrida {i+1}/{N_CORRIDAS} | seed={seed}")
    print(f"{'='*50}")

    for target_attr in TARGET_ATTRS:
        seeds_usadas[target_attr].append(seed)
        config = {**BASE_CONFIG, "target_attr": target_attr, "seed": seed,
                  "model_save_path": f"models/simple_{target_attr}_corrida{i+1}_seed{seed}.pth"}

        print(f"\n  Target: {target_attr}")
        train_loader, val_loader, test_loader, dataset, test_df = get_dataloaders_simple(config)
        model = train(config, train_loader, val_loader, device)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images).cpu().numpy().flatten()
                all_preds.extend((outputs > 0.5).astype(int))
                all_labels.extend(labels.numpy())
        acc = accuracy_score(all_labels, all_preds)
        results[target_attr].append(acc)
        print(f"  Test acc: {acc:.4f}")

print("\nResumen:")
for target_attr in TARGET_ATTRS:
    accs = results[target_attr]
    print(f"  {target_attr}: mean={np.mean(accs):.4f} | std={np.std(accs):.4f}")

os.makedirs("results", exist_ok=True)
with open("results/results_simple.pkl", "wb") as f:
    pickle.dump({"results": results, "seeds": seeds_usadas}, f)
print("\nResultados guardados en results/results_simple.pkl")
