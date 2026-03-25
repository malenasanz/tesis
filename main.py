from src.dataloader import get_dataloaders
import torch
import torch.nn as nn
from src.model import SimpleCNN
from src.train import train
from src.eval import evaluate

config = {
    "img_dir": "data/raw/celeba/img_align_celeba",
    "attr_path": "data/raw/celeba/list_attr_celeba.txt",
    "target_attr": "Smiling",
    "batch_size": 64,
    "num_epochs": 5,
    "learning_rate": 0.001,
    "model_save_path": "models/simplecnn.pth",
    "subset_size": 8000  # Set to None to use the full dataset
}

# Get dataloaders
train_loader, test_loader = get_dataloaders(config)
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of test batches: {len(test_loader)}")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model, optimizer, criterion
model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
criterion = nn.BCELoss()

# Training loop
for epoch in range(config["num_epochs"]):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    test_acc = evaluate(model, test_loader, device)
    
    print(f"Epoch {epoch+1}/{config['num_epochs']}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Save the model
torch.save(model.state_dict(), config["model_save_path"])
print(f"Model saved to {config['model_save_path']}")

