import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        sequences = batch["seq"].to(device)
        labels = batch["labels"].to(device)

        # Forward
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            sequences = batch["seq"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train_val(
    epochs,
    model,
    train_dataloader,
    test_dataloader,
    criterion,
    optimizer,
    device,
    scheduler,
    show_plot,
    save_model_dir,
    model_name,
    pretrained_param_dir=None,
    print_interval=20,
):
    if pretrained_param_dir:
        model = model.load_state_dict(torch.load(pretrained_param_dir))
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    print("Start training model...")
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        train_loss, train_acc = train_epoch(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, test_dataloader, criterion, device)

        scheduler.step()

        if epoch % print_interval == 0:
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model_path = os.path.join(save_model_dir, model_name + ".pth")
            torch.save(model.state_dict(), save_model_path)
            print("✓ Saved best model")

    if show_plot:
        plt.plot(train_losses)
        plt.plot(val_losses)
