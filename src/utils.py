"""
Utility functions for ViT training pipeline.

Includes: data loading, training loop, validation, logging, metrics,
and reproducibility helpers.
"""

import time
import random
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .model import VisionTransformer, count_parameters


def get_device() -> torch.device:
    """Return best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    augment_train: bool = True,
    img_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 train and test dataloaders.

    Args:
        data_dir: Directory to store/load CIFAR-10
        batch_size: Batch size for both loaders
        num_workers: DataLoader workers
        augment_train: Whether to use augmentations for training
        img_size: Output image size (32 for native CIFAR, 224 for timm ViT)

    Returns:
        (train_loader, test_loader)
    """
    if img_size == 32:
        # CIFAR-10 native normalization
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        )
    else:
        # ImageNet normalization (for timm pretrained ViT)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    resize = transforms.Resize((img_size, img_size))

    if augment_train:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                resize,
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                resize,
                transforms.ToTensor(),
                normalize,
            ]
        )

    test_transform = transforms.Compose(
        [
            resize,
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # pin_memory speeds up CPU->GPU transfer but MPS doesn't support it
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader


def get_pretrained_vit(
    model_name: str = "vit_tiny_patch16_224",
    num_classes: int = 10,
    pretrained: bool = True,
) -> nn.Module:
    """Load pre-trained ViT from timm for CIFAR-10 fine-tuning.

    Uses ImageNet-pretrained weights; expects 224x224 input (CIFAR resized).
    """
    import timm

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def get_model(
    img_size: int = 32,
    patch_size: int = 4,
    embed_dim: int = 192,
    depth: int = 6,
    num_heads: int = 6,
    num_classes: int = 10,
    dropout: float = 0.0,
) -> VisionTransformer:
    """Create ViT model with standard CIFAR-10 config."""
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        dropout=dropout,
    )


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float, float]:
    """
    Train for one epoch. Returns (avg_loss, accuracy, images_per_sec).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.perf_counter()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    elapsed = time.perf_counter() - start_time
    n_images = total
    throughput = n_images / elapsed if elapsed > 0 else 0.0
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, throughput


@torch.no_grad()
def validate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def get_gpu_memory_mb() -> Optional[float]:
    """Return current GPU memory allocated in MB, or None if no GPU."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return None


def get_peak_gpu_memory_mb() -> Optional[float]:
    """Return peak GPU memory allocated in MB, or None if no GPU."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024**2
    return None


def reset_peak_gpu_memory() -> None:
    """Reset peak memory stats for fresh measurement."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 3e-4,
    device: Optional[torch.device] = None,
    log_every: int = 1,
    target_accuracy: Optional[float] = 94.0,
) -> Dict[str, Any]:
    """
    Full training loop with optional early stopping at target accuracy.

    Args:
        target_accuracy: If set, stop when test accuracy >= this value.
            Primary metric is time_to_target (wall-clock seconds to reach it).

    Returns:
        Dictionary with 'history', 'best_acc', 'total_time', 'time_to_target' (sec to hit 94%, or None).
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "throughput": [],
        "wall_time": [],
    }
    best_acc = 0.0
    time_to_target: Optional[float] = None
    total_start = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc, throughput = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        epoch_time = time.perf_counter() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["throughput"].append(throughput)
        history["wall_time"].append(epoch_time)

        best_acc = max(best_acc, test_acc)

        # Record time when target accuracy is first reached
        if target_accuracy is not None and time_to_target is None and test_acc >= target_accuracy:
            time_to_target = time.perf_counter() - total_start

        if log_every > 0 and epoch % log_every == 0:
            peak_mem = get_peak_gpu_memory_mb()
            mem_str = f"{peak_mem:.1f} MB" if peak_mem else "N/A"
            target_str = f" [TARGET HIT @ {time_to_target:.1f}s]" if time_to_target is not None else ""
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
                f"Throughput: {throughput:.0f} img/s | "
                f"Mem: {mem_str}{target_str}"
            )

        # Early stop when target reached
        if target_accuracy is not None and test_acc >= target_accuracy:
            print(f"\nTarget {target_accuracy}% reached at epoch {epoch}. Time-to-target: {time_to_target:.2f}s")
            break

    total_time = time.perf_counter() - total_start
    return {
        "history": history,
        "best_acc": best_acc,
        "total_time": total_time,
        "time_to_target": time_to_target,
        "peak_gpu_mb": get_peak_gpu_memory_mb(),
    }


def setup_logging(log_path: Optional[Path] = None) -> None:
    """Configure logging to console and optionally to file."""
    handlers = [logging.StreamHandler()]
    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )
