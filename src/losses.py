"""
Loss functions for ViT knowledge distillation experiments.

Includes:
- Baseline cross-entropy (CE)
- Vanilla Knowledge Distillation (KD)
- Decoupled Knowledge Distillation (DKD)
- DKD + Feature Distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_ce(zs: torch.Tensor, y: torch.Tensor, label_smoothing: float = 0.1) -> torch.Tensor:
    """
    Standard cross-entropy loss with optional label smoothing.

    Args:
        zs: Student logits (B, num_classes)
        y: Ground-truth labels (B,)
        label_smoothing: Label smoothing factor (default 0.1)
    """
    return F.cross_entropy(zs, y, label_smoothing=label_smoothing)


def loss_vanilla_kd(
    zs: torch.Tensor,
    zt: torch.Tensor,
    y: torch.Tensor,
    T: float = 4.0,
    kd_weight: float = 4.0,
) -> torch.Tensor:
    """
    Vanilla Knowledge Distillation loss.

    Combines standard cross-entropy with KL divergence between
    student and teacher softened probability distributions.

    Args:
        zs: Student logits (B, num_classes)
        zt: Teacher logits (B, num_classes)
        y: Ground-truth labels (B,)
        T: Distillation temperature
        kd_weight: Weight on distillation loss term
    """
    ce = F.cross_entropy(zs, y)
    kd = F.kl_div(
        F.log_softmax(zs / T, dim=1),
        F.softmax(zt / T, dim=1),
        reduction="batchmean",
    ) * (T * T)
    return ce + kd_weight * kd


def _gt_mask(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Create boolean mask for ground-truth class positions.

    Args:
        z: Logits (B, num_classes)
        y: Ground-truth labels (B,)
    """
    return torch.zeros_like(z).scatter_(1, y.view(-1, 1), 1).bool()


def loss_dkd(
    zs: torch.Tensor,
    zt: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 8.0,
    T: float = 4.0,
) -> torch.Tensor:
    """
    Decoupled Knowledge Distillation (DKD) loss.

    Separates distillation into:
    - TCKD: target class knowledge distillation
    - NCKD: non-target dark knowledge distillation

    Loss = CE + alpha * TCKD + beta * NCKD

    Args:
        zs: Student logits (B, num_classes)
        zt: Teacher logits (B, num_classes)
        y: Ground-truth labels (B,)
        alpha: Weight on target-class distillation (TCKD)
        beta: Weight on non-target distillation (NCKD)
        T: Distillation temperature
    """
    gt = _gt_mask(zs, y)
    other = ~gt

    ps = F.softmax(zs / T, dim=1)
    pt = F.softmax(zt / T, dim=1)

    # TCKD — 2-bin KL on [P(target), P(rest)]
    ps2 = torch.cat([(ps * gt).sum(1, keepdim=True), (ps * other).sum(1, keepdim=True)], 1)
    pt2 = torch.cat([(pt * gt).sum(1, keepdim=True), (pt * other).sum(1, keepdim=True)], 1)
    tckd = F.kl_div(ps2.clamp_min(1e-12).log(), pt2, reduction="sum") * T * T / y.shape[0]

    # NCKD — KL over non-target classes only
    pt_nt = F.softmax(zt / T - 1000.0 * gt.float(), dim=1)
    ps_nt = F.log_softmax(zs / T - 1000.0 * gt.float(), dim=1)
    nckd = F.kl_div(ps_nt, pt_nt, reduction="sum") * T * T / y.shape[0]

    ce = F.cross_entropy(zs, y)
    return ce + alpha * tckd + beta * nckd


def loss_dkd_plus_feature(
    zs: torch.Tensor,
    zt: torch.Tensor,
    y: torch.Tensor,
    s_feat: torch.Tensor,
    t_feat: torch.Tensor,
    proj: torch.nn.Module,
    alpha: float = 1.0,
    beta: float = 8.0,
    T: float = 4.0,
    feat_weight: float = 4.0,
) -> torch.Tensor:
    """
    DKD + Feature Distillation loss.

    Combines DKD with MSE alignment between projected student
    CLS token and teacher penultimate features.

    Loss = DKD + feat_weight * MSE(proj(student_feat), teacher_feat)

    Args:
        zs: Student logits (B, num_classes)
        zt: Teacher logits (B, num_classes)
        y: Ground-truth labels (B,)
        s_feat: Student CLS token features (B, student_dim)
        t_feat: Teacher penultimate features (B, teacher_dim)
        proj: Learnable linear projection student_dim -> teacher_dim
        alpha: TCKD weight
        beta: NCKD weight
        T: Distillation temperature
        feat_weight: Weight on feature alignment loss
    """
    base = loss_dkd(zs, zt, y, alpha=alpha, beta=beta, T=T)
    feat = F.mse_loss(proj(s_feat), t_feat.flatten(1).detach())
    return base + feat_weight * feat
