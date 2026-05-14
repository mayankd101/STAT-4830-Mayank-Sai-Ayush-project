# Self-Critique: Week 10 Vanilla Knowledge Distillation

## OBSERVE

After reviewing the Week 10 implementation, the teacher-student distillation framework is functional and integrates correctly with the existing systems-level optimizations. The ResNet-56 teacher is successfully frozen and produces stable soft targets throughout training. The student ViT trains using a weighted combination of cross-entropy and KL divergence losses without introducing numerical instability. Validation curves show smoother convergence compared to baseline training. However, Vanilla KD does not substantially improve time-to-target performance over the optimized baseline, suggesting the distillation objective itself is the limiting factor rather than hardware efficiency.

## ORIENT

### Strengths
- Teacher-student framework integrates cleanly with existing BF16, AdamW, and cosine scheduling pipeline
- Smoother validation curves and reduced oscillation compared to baseline ViT
- Soft targets from ResNet-56 successfully transfer inter-class relationship information

### Areas for Improvement
- Vanilla KD treats all non-target classes equally, limiting the quality of relational information transferred
- Time-to-target improvement over optimized baseline is minimal — convergence bottleneck has shifted from hardware to distillation objective structure
- Temperature and weighting hyperparameters (α, T) have not been systematically tuned

### Critical Risks/Assumptions
- Results assume ResNet-56 teacher is well-calibrated and its soft targets are informative for CIFAR-10
- Improvements in convergence smoothness do not necessarily translate into lower time-to-target
- Single-seed experiments limit confidence in reproducibility of observed gains

## DECIDE

### Concrete Next Actions
- Implement Decoupled Knowledge Distillation (DKD) to separately control target-class and non-target-class supervision
- Tune temperature T and weighting α systematically across controlled experiments
- Measure exact wall-clock improvements from KD variants relative to baseline
- Analyze whether feature-level supervision can provide stronger optimization signal than logits alone

## ACT

### Resource Needs
- Additional GPU compute time for systematic hyperparameter sweeps over α and T
- Reference implementation of DKD from Megvii mdistiller repository for validation
- Structured logging to track per-epoch convergence speed across distillation configurations
