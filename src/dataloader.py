from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset
from src.datasets import CelebADataSet
import numpy as np


def get_dataloaders(config):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = CelebADataSet(
        img_dir=config["img_dir"],
        attr_path=config["attr_path"],
        transform=transform,
        target_attr=config["target_attr"]
    )

    if config["subset_size"]:
        indices = np.random.choice(len(dataset), config["subset_size"], replace=False)
        dataset = Subset(dataset, indices)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    return train_loader, test_loader