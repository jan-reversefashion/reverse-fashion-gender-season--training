"""Train a ResNet-50 classifier on a local ImageFolder dataset.

Includes automatic cyclic LR range finding and logs loss/speed every 100 batches.

Usage:
    uv run scripts/train.py --data /home/sagemaker-user/user-default-efs/data/sellpy2
    uv run scripts/train.py --data /home/sagemaker-user/user-default-efs/data/sellpy2 --epochs 15 --batch-size 64
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-50 on ImageFolder dataset")
    parser.add_argument("--data", default="/home/sagemaker-user/user-default-efs/data/sellpy2", help="ImageFolder root")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr-find-batches", type=int, default=100, help="Batches for LR range test")
    parser.add_argument("--output", default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_transforms(image_size, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def lr_range_test(model, loader, criterion, device, num_batches=100, start_lr=1e-7, end_lr=1e-1):
    """Run an LR range test and return suggested base_lr and max_lr for cyclic schedule."""
    optimizer = optim.SGD(model.parameters(), lr=start_lr, momentum=0.9)
    lr_mult = (end_lr / start_lr) ** (1 / num_batches)

    lrs, losses = [], []
    running_loss = 0.0
    best_loss = float("inf")
    batch_iter = iter(loader)

    model.train()
    for i in range(num_batches):
        try:
            images, labels = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            images, labels = next(batch_iter)

        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        smoothed = 0.98 * running_loss + 0.02 * loss.item() if i > 0 else loss.item()
        running_loss = smoothed
        corrected = smoothed / (1 - 0.98 ** (i + 1))

        lrs.append(optimizer.param_groups[0]["lr"])
        losses.append(corrected)

        if corrected < best_loss:
            best_loss = corrected

        # Stop if loss diverges
        if corrected > best_loss * 4:
            break

        # Increase LR
        for pg in optimizer.param_groups:
            pg["lr"] *= lr_mult

    # Find the LR with steepest loss decrease
    min_loss_idx = losses.index(min(losses))
    max_lr = lrs[min_loss_idx]
    base_lr = max_lr / 10

    return base_lr, max_lr


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, num_epochs, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start = time.time()
    batch_start = time.time()
    use_amp = device.type == "cuda"

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - batch_start
            imgs_per_sec = 100 * images.size(0) / elapsed
            avg_loss = running_loss / total
            print(
                f"  [{epoch+1}/{num_epochs}] batch {batch_idx+1}/{len(loader)} | "
                f"loss: {avg_loss:.4f} | acc: {correct/total:.4f} | "
                f"{imgs_per_sec:.0f} img/s | lr: {optimizer.param_groups[0]['lr']:.2e}"
            )
            batch_start = time.time()

    epoch_time = time.time() - epoch_start
    return running_loss / total, correct / total, epoch_time


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required but not available")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    # Load dataset
    data_dir = Path(args.data)
    full_dataset = datasets.ImageFolder(data_dir, transform=get_transforms(args.image_size, train=True))
    num_classes = len(full_dataset.classes)
    print(f"Dataset: {len(full_dataset)} images, {num_classes} classes: {full_dataset.classes}")

    # Train/val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )
    # Use a separate dataset with val transforms so train augmentations are not overwritten
    val_full_dataset = datasets.ImageFolder(data_dir, transform=get_transforms(args.image_size, train=False))
    val_dataset = torch.utils.data.Subset(val_full_dataset, val_dataset.indices)

    pin = torch.cuda.is_available()
    persistent = args.workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin, persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin, persistent_workers=persistent,
    )
    print(f"Train: {train_size}, Val: {val_size}")

    # Build model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Auto-tune cyclic LR via range test
    print("\nRunning LR range test...")
    base_lr, max_lr = lr_range_test(
        model, train_loader, criterion, device, num_batches=args.lr_find_batches
    )
    print(f"Auto-tuned cyclic LR: base_lr={base_lr:.2e}, max_lr={max_lr:.2e}")

    # Re-init model (range test corrupted weights)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    model = torch.compile(model)

    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=2 * len(train_loader),
        mode="triangular2",
        cycle_momentum=False,
    )

    # Train
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Cyclic LR: base={base_lr:.2e}, max={max_lr:.2e}, step_size_up={2*len(train_loader)}\n")

    for epoch in range(args.epochs):
        train_loss, train_acc, epoch_time = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, args.epochs, scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{args.epochs} done in {epoch_time:.1f}s | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": full_dataset.classes,
                "base_lr": base_lr,
                "max_lr": max_lr,
            }, ckpt_path)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")
    print(f"Checkpoint: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
