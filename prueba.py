from src.dataloader import get_dataloaders
from src.eval import evaluate
from src.model import SimpleCNN
import torch

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = get_dataloaders(config)
model = SimpleCNN()
model.load_state_dict(torch.load(config["model_save_path"], map_location=device))
model.to(device)
acc = evaluate(model, test_loader, device)
print(acc)