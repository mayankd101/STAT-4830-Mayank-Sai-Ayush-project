# Week 12: Decoupled Knowledge Distillation and Convergence Optimization

## Problem Statement

Previous experiments demonstrated that Vanilla Knowledge Distillation improves convergence smoothness but provides limited gains in time-to-target performance. One major limitation of standard KD is that it treats all non-target classes uniformly, preventing the student model from fully leveraging the relational information encoded by the teacher network.

The goal of this stage of the project is to improve convergence efficiency by implementing Decoupled Knowledge Distillation (DKD). The objective is to determine whether separating target-class supervision from non-target-class supervision can accelerate Vision Transformer learning on CIFAR-10.

This phase focuses specifically on improving optimization dynamics rather than computational throughput. The central metric remains wall-clock time required to reach 85% test accuracy.

---

## Technical Approach

The project extends the previous teacher–student framework using Decoupled Knowledge Distillation.

Instead of using a single distillation term, DKD separates the learning objective into:
- TCKD (Target Class Knowledge Distillation)
- NCKD (Non-Target Class Knowledge Distillation)

The overall loss is written as:

$$
L = L_{CE} + \alpha L_{TCKD} + \beta L_{NCKD}
$$

where:
- $\alpha$ controls target-class alignment
- $\beta$ controls non-target relational supervision

This decomposition allows the student model to independently learn:
1. correctness of predictions
2. inter-class relationships encoded by the teacher

The teacher remains a frozen ResNet-56 network, while the student is the custom Vision Transformer architecture developed earlier in the project.

Several hyperparameter studies are conducted to evaluate:
- distillation temperature
- target vs non-target weighting
- learning-rate sensitivity
- stability under aggressive optimization settings

The implementation continues to use:
- BF16 mixed precision
- cosine learning-rate scheduling
- warmup scheduling
- optimized PyTorch training loops

Time-to-target measurements are collected for all experimental configurations.

---

## Initial Results

The DKD implementation produces a significant reduction in wall-clock time relative to both the baseline ViT and Vanilla KD experiments.

Training curves indicate:
- faster early-stage convergence
- smoother optimization behavior
- improved stability at higher learning rates

The separation between target and non-target supervision appears to improve how effectively the ViT learns from teacher outputs. The student reaches 85% test accuracy substantially earlier than previous methods.

Experimental results also suggest that non-target supervision contributes strongly to convergence speed. Preserving class similarity structure allows the ViT to learn more informative representations earlier during training.

One challenge encountered during this stage is increased hyperparameter sensitivity. The balance between target and non-target losses significantly affects convergence quality, requiring careful tuning of $\alpha$, $\beta$, and temperature values.

Despite the additional tuning complexity, DKD demonstrates clear improvements in learning efficiency over prior approaches.

---

## Next Steps

The next phase of the project is to investigate feature-level distillation in addition to output-level distillation.

The hypothesis is that matching intermediate representations between the teacher and student networks can provide stronger supervision signals than logits alone.

Planned work includes:
- implementing feature alignment losses
- selecting appropriate transformer layers for alignment
- measuring additional convergence gains
- evaluating computational overhead introduced by feature matching

The project will also continue refining DKD hyperparameters and profiling the relationship between convergence speed and representation quality.

The primary lesson learned during this stage is that improving optimization objectives can reduce training time more effectively than throughput optimizations alone.
