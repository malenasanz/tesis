from src.dataloader import get_dataloaders
import torch
import torch.nn as nn
from src.model import SimpleCNN
from src.train import train
from src.eval import evaluate, evaluate_subgroups

config_balanceado = {
    "img_dir": "data/raw/celeba/img_align_celeba",
    "attr_path": "data/raw/celeba/list_attr_celeba.txt",
    "target_attr": "Blond_Hair",
    "bias_attr": "Male",
    "balanced": True,
    "batch_size": 64,
    "num_epochs": 5,
    "learning_rate": 0.001,
    "model_save_path": "models/menos_mujeres_rubias.pth",
}

config = {
    "img_dir": "data/raw/celeba/img_align_celeba",
    "attr_path": "data/raw/celeba/list_attr_celeba.txt",
    "target_attr": "Blond_Hair",
    "bias_attr": "Male",
    "proportions": {
    (0, 0): 0.20,  # mujer no rubia  (más)
    (0, 1): 0.20,  # mujer rubia     (menos)
    (1, 0): 0.20,  # hombre no rubio
    (1, 1): 0.40  # hombre rubio
    },
"n_total": 4300,
    "batch_size": 64,
    "num_epochs": 5,
    "learning_rate": 0.001,
    "model_save_path": "models/mas_hombres_rubios.pth",
}



# Get dataloaders
train_loader, val_loader, test_loader, dataset, test_df = get_dataloaders(config)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model, optimizer, criterion
model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
criterion = nn.BCELoss()

# Training loop
for epoch in range(config["num_epochs"]):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}/{config['num_epochs']} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Evaluate on test set by subgroup
print("\nTest evaluation by subgroup:")
subgroup_results = evaluate_subgroups(model, dataset, test_df, config["bias_attr"], config["target_attr"], device)
for (v1, v2), metrics in subgroup_results.items():
    print(f"  {config['bias_attr']}={v1}, {config['target_attr']}={v2} | Acc: {metrics['acc']:.4f}, n={metrics['n']}")

# Save the model and subgroup accuracies
torch.save({
    "model_state_dict": model.state_dict(),
    "config": config,
    "subgroup_accuracies": {str(k): v["acc"] for k, v in subgroup_results.items()},
}, config["model_save_path"])
print(f"Model saved to {config['model_save_path']}")

