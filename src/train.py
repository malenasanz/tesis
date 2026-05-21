import torch
from sklearn.metrics import accuracy_score
from .eval import evaluate


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    multiclass = isinstance(criterion, torch.nn.CrossEntropyLoss)
    for images, labels in dataloader:
        images = images.to(device)
        labels_device = labels.long().to(device) if multiclass else labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1) if multiclass else (outputs > 0).squeeze(1)
        all_preds.extend(preds.cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds)


def train(config, train_loader, val_loader, device):
    model_kwargs = {k: config[k] for k in ("n_capas",) if k in config}
    model     = config["model_class"](**model_kwargs).to(device)
    optimizer = config["optimizer_class"](model.parameters(), lr=config["learning_rate"])
    criterion = config["criterion"]
    patience  = config.get("patience", None)

    best_val_loss   = float("inf")
    best_weights    = None
    epochs_no_improve = 0

    for epoch in range(config["num_epochs"]):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"  Epoch {epoch+1}/{config['num_epochs']} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if config.get("save_each_epoch") and config.get("model_save_path"):
            epoch_path = config["model_save_path"].replace(".pth", f"_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), epoch_path)

        if patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"  Early stopping en epoch {epoch+1} (sin mejora por {patience} epochs)")
                    model.load_state_dict(best_weights)
                    break

    if config.get("model_save_path"):
        torch.save(model.state_dict(), config["model_save_path"])
        print(f"  Modelo guardado en {config['model_save_path']}")

    return model
