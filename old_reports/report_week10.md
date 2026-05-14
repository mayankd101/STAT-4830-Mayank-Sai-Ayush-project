# Week 10: Transition from System Optimization to Knowledge Distillation

## Problem Statement

Although the previous system-level optimizations improved throughput and reduced GPU memory usage, the Vision Transformer still required substantial wall-clock time to reach the target accuracy threshold on CIFAR-10. Profiling experiments revealed that improvements in per-step computational efficiency did not necessarily translate into proportional improvements in convergence speed. The primary bottleneck appeared to be the learning dynamics of the model itself rather than raw hardware utilization.

The objective of this phase of the project is therefore to investigate whether knowledge distillation can improve learning efficiency and reduce the time required for a Vision Transformer (ViT) to achieve 85% test accuracy on CIFAR-10. Instead of training exclusively from one-hot labels, the student ViT is trained using supervision from a stronger convolutional teacher network.

This shift is motivated by the observation that convolutional neural networks (CNNs) converge significantly faster on small datasets such as CIFAR-10 due to stronger inductive biases. By transferring information from a pretrained CNN teacher into the ViT student, the project aims to retain the representational flexibility of transformers while improving convergence behavior.

The primary evaluation metric remains time-to-target accuracy. Secondary metrics include training stability, validation accuracy progression, throughput, and the smoothness of optimization dynamics.

---

## Technical Approach

The project now adopts a teacher–student distillation framework. A pretrained ResNet-56 model is used as the teacher network, while the Vision Transformer remains the student model.

The baseline distillation method implemented is Vanilla Knowledge Distillation (KD). The student is optimized using a weighted combination of standard cross-entropy loss and a distillation loss between the teacher and student logits:

$$
L = \alpha L_{CE} + (1-\alpha)L_{KD}
$$

where $L_{CE}$ represents cross-entropy with hard labels and $L_{KD}$ represents the divergence between softened teacher and student probability distributions.

The softened probabilities are computed using temperature scaling:

$$
p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$

where $T$ is the distillation temperature and $z_i$ denotes logits.

The intuition behind distillation is that teacher probability distributions encode “dark knowledge” about inter-class similarity. Instead of learning only the correct class label, the ViT can learn richer relationships between classes, potentially improving optimization efficiency and convergence speed.

The implementation integrates the previously developed systems-level optimizations:
- BF16 automatic mixed precision
- AdamW optimization with cosine learning-rate scheduling
- warmup scheduling
- optimized data loading pipelines

The project additionally introduces controlled experiments comparing:
- baseline ViT training
- system-optimized ViT training
- ViT training with Vanilla KD

---

## Initial Results

Initial experiments demonstrate that knowledge distillation produces smoother and more stable convergence behavior compared to the baseline Vision Transformer.

The student ViT successfully learns from the ResNet-56 teacher outputs without introducing instability into the training pipeline. Validation curves exhibit reduced oscillation, and optimization appears more consistent across epochs.

However, preliminary results indicate that Vanilla KD alone does not substantially outperform the optimized baseline in terms of time-to-target performance. Although convergence becomes smoother, the student still treats all non-target classes similarly, limiting the amount of useful relational information transferred from the teacher.

Profiling results suggest that the bottleneck has shifted from hardware efficiency toward the structure of the distillation objective itself. This indicates that more advanced distillation strategies may be necessary to further accelerate convergence.

---

## Next Steps

The next phase of the project is to investigate Decoupled Knowledge Distillation (DKD), which separates target-class and non-target-class supervision into distinct objectives.

The main hypothesis is that explicitly separating:
- target class alignment
- non-target “dark knowledge”

will provide finer control over learning dynamics and improve convergence speed relative to Vanilla KD.

Additional goals include:
- tuning temperature and weighting hyperparameters
- measuring exact wall-clock improvements from KD variants
- analyzing convergence stability under different distillation objectives
- determining whether feature-level supervision can further accelerate training

The most important insight from this stage is that faster training requires improvements not only in computation speed but also in how efficiently information is learned during optimization.
