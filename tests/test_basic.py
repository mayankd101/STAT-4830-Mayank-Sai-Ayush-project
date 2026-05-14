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
    train,
    validate,
    get_peak_gpu_memory_mb,
)
from src.losses import loss_ce, loss_vanilla_kd, loss_dkd, loss_dkd_plus_feature


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


def test_set_seed_deterministic_false():
    """set_seed with deterministic=False does not raise."""
    set_seed(42, deterministic=False)


def test_get_cifar10_loaders_with_randaugment():
    """DataLoader works with use_randaugment=True."""
    set_seed(42)
    train_loader, test_loader = get_cifar10_loaders(
        data_dir="./data",
        batch_size=32,
        num_workers=0,
        augment_train=True,
        use_randaugment=True,
    )
    batch, _ = next(iter(train_loader))
    assert batch.shape[0] == 32
    assert batch.shape[1] == 3


def test_train_with_amp_and_scheduler(model, device):
    """train() runs with use_amp and use_scheduler (2 epochs)."""
    set_seed(42)
    train_loader, test_loader = get_cifar10_loaders(
        data_dir="./data",
        batch_size=32,
        num_workers=0,
        augment_train=True,
    )
    model = model.to(device)
    results = train(
        model,
        train_loader,
        test_loader,
        num_epochs=2,
        lr=3e-4,
        device=device,
        log_every=2,
        target_accuracy=None,
        use_amp=True,
        use_scheduler=True,
    )
    assert "history" in results
    assert len(results["history"]["test_acc"]) == 2


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


def test_loss_ce(model, device):
    """CE loss returns positive scalar."""
    model = model.to(device)
    x = torch.randn(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    logits = model(x)
    loss = loss_ce(logits, y)
    assert loss.item() > 0


def test_loss_vanilla_kd(model, device):
    """Vanilla KD loss returns positive scalar."""
    model = model.to(device)
    x = torch.randn(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    zs = model(x)
    zt = torch.randn_like(zs)
    loss = loss_vanilla_kd(zs, zt, y)
    assert loss.item() > 0


def test_loss_dkd(model, device):
    """DKD loss returns positive scalar."""
    model = model.to(device)
    x = torch.randn(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    zs = model(x)
    zt = torch.randn_like(zs)
    loss = loss_dkd(zs, zt, y)
    assert loss.item() > 0


def test_loss_dkd_plus_feature(model, device):
    """DKD + feature loss returns positive scalar."""
    model = model.to(device)
    x = torch.randn(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    zs = model(x)
    zt = torch.randn_like(zs)
    s_feat = torch.randn(4, 192, device=device)
    t_feat = torch.randn(4, 64, device=device)
    proj = torch.nn.Linear(192, 64, bias=False).to(device)
    loss = loss_dkd_plus_feature(zs, zt, y, s_feat, t_feat, proj)
    assert loss.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
