from sklearn.metrics import accuracy_score
from src.eval import evaluate


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend((outputs > 0.5).cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds)


def train(config, train_loader, val_loader, device):
    model     = config["model_class"]().to(device)
    optimizer = config["optimizer_class"](model.parameters(), lr=config["learning_rate"])
    criterion = config["criterion"]

    for epoch in range(config["num_epochs"]):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"  Epoch {epoch+1}/{config['num_epochs']} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return model
