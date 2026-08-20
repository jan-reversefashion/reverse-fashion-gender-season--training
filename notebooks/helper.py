import boto3
import torch
import torch.nn as nn
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


class S3ImageDataset(Dataset):
    """PyTorch Dataset that reads images directly from S3."""

    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (s3_path, label_idx)
        self.transform = transform
        self._s3 = None

    @property
    def s3(self):
        # Lazy init so each DataLoader worker gets its own client (fork-safe)
        if self._s3 is None:
            self._s3 = boto3.client("s3")
        return self._s3

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s3_path, label = self.samples[idx]
        parts = s3_path.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        response = self.s3.get_object(Bucket=bucket, Key=key)
        img = Image.open(BytesIO(response["Body"].read())).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(image_size, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_dataloaders(sample_data, val_split, image_size, batch_size, num_workers=4):
    total = len(sample_data)
    val_size = int(total * val_split)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total, generator=generator).tolist()
    train_samples = [sample_data[i] for i in indices[:train_size]]
    val_samples = [sample_data[i] for i in indices[train_size:]]

    train_dataset = S3ImageDataset(train_samples, transform=get_transforms(image_size, train=True))
    val_dataset = S3ImageDataset(val_samples, transform=get_transforms(image_size, train=False))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, train_size, val_size


def build_resnet50(num_classes, device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          device, num_epochs, class_to_idx, target_field, save_dir="."):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            path = f"{save_dir}/best_resnet50_{target_field}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_to_idx": class_to_idx,
                "target_field": target_field,
            }, path)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    return best_val_acc
