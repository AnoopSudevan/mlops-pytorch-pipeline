import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import get_model


def parse_args():
    p = argparse.ArgumentParser(description="Train CIFAR-10 image classifier")
    p.add_argument("--config", default=None, help="Path to training_config.yaml")
    return p.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total


def main():
    args = parse_args()

    # resolve config path: CLI arg > docker default > local fallback
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path("/app/configs/training_config.yaml")
        if not config_path.exists():
            config_path = Path("configs/training_config.yaml")

    if not config_path.exists():
        print(json.dumps({"event": "error", "message": f"Config not found: {config_path}"}))
        sys.exit(1)

    config = load_config(str(config_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "startup", "device": str(device)}), flush=True)

    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        dataset=config["data"].get("dataset", "cifar10"),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )
    # cosine annealing tends to work well without much tuning
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        log = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "lr": round(scheduler.get_last_lr()[0], 6),
        }
        print(json.dumps(log), flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            save_path = checkpoint_dir / config["output"]["model_name"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "config": config,
            }, save_path)
            print(json.dumps({"event": "checkpoint_saved", "path": str(save_path)}), flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(json.dumps({"event": "early_stopping", "epoch": epoch + 1}), flush=True)
                break

    print(json.dumps({"event": "training_complete", "best_val_loss": round(best_val_loss, 4)}), flush=True)


if __name__ == "__main__":
    main()
