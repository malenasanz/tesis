import torch
from sklearn.metrics import accuracy_score

def evaluate(model, dataloader, device):
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            outputs = model(images)
            preds = (outputs > 0.5).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return accuracy_score(all_labels, all_preds)