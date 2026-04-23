import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score

def evaluate(model, dataloader, criterion, device):
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels_device = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels_device)
            total_loss += loss.item()

            preds = (outputs > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), test_acc


def evaluate_subgroups(model, dataset, test_df, bias_attr, target_attr, device, batch_size=64):
    name_to_idx = {name: i for i, name in enumerate(dataset.images)}
    results = {}

    for v1 in [0, 1]:
        for v2 in [0, 1]:
            group_df = test_df[(test_df[bias_attr] == v1) & (test_df[target_attr] == v2)]
            indices = [name_to_idx[n] for n in group_df.index]
            loader = DataLoader(Subset(dataset, indices), batch_size=batch_size)
            _, acc = evaluate(model, loader, nn.BCELoss(), device)
            results[(v1, v2)] = {"acc": acc, "n": len(indices)}

    return results