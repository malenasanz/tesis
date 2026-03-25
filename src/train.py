import torch
import torch.nn as nn
from src.dataloader import get_dataloaders
from src.model import SimpleCNN

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)