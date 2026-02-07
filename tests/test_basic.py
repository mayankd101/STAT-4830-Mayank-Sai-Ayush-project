"""
Basic validation tests for ViT training pipeline.

Run with: pytest tests/test_basic.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import VisionTransformer, count_parameters
from src.utils import (
    set_seed,
    get_cifar10_loaders,
    get_model,
    train_one_epoch,
    validate,
    get_peak_gpu_memory_mb,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model():
    set_seed(42)
    return get_model(
        patch_size=4,
        embed_dim=192,
        depth=6,
        num_heads=6,
        dropout=0.0,
    )


def test_model_forward_shape(model, device):
    """Model outputs (B, 10) logits for CIFAR-10."""
    set_seed(42)
    model = model.to(device)
    x = torch.randn(4, 3, 32, 32, device=device)
    out = model(x)
    assert out.shape == (4, 10)


def test_model_parameter_count(model):
    """Model has reasonable number of parameters."""
    n = count_parameters(model)
    assert 100_000 < n < 50_000_000


def test_train_one_epoch(model, device):
    """Training one epoch runs without error and returns valid metrics."""
    set_seed(42)
    train_loader, _ = get_cifar10_loaders(
        data_dir="./data",
        batch_size=32,
        num_workers=0,  # Avoid multiprocessing in tests
        augment_train=True,
    )
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    loss, acc, throughput = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch=1
    )

    assert loss > 0
    assert 0 <= acc <= 100
    assert throughput > 0


def test_validate(model, device):
    """Validation runs without error and returns valid metrics."""
    set_seed(42)
    _, test_loader = get_cifar10_loaders(
        data_dir="./data",
        batch_size=32,
        num_workers=0,
        augment_train=False,
    )
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    loss, acc = validate(model, test_loader, criterion, device)

    assert loss > 0
    assert 0 <= acc <= 100


def test_deterministic_with_seed(model, device):
    """Same seed produces same initial forward pass output."""
    set_seed(42)
    m1 = get_model(patch_size=4, embed_dim=192, depth=6, num_heads=6)
    set_seed(42)
    m2 = get_model(patch_size=4, embed_dim=192, depth=6, num_heads=6)

    x = torch.randn(2, 3, 32, 32, device=device)
    m1, m2 = m1.to(device), m2.to(device)

    with torch.no_grad():
        out1 = m1(x)
        out2 = m2(x)

    torch.testing.assert_close(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
