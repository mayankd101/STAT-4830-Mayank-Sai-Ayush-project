# Self-Critique: Week 12 Decoupled Knowledge Distillation

## OBSERVE

After reviewing the Week 12 implementation, DKD produces a significant and measurable reduction in wall-clock time to 85% test accuracy relative to both the baseline ViT and Vanilla KD. The separation of target-class and non-target-class supervision objectives improves early-stage convergence speed and optimization stability. Training curves are smoother and reach the target threshold substantially earlier than prior methods. However, DKD introduces increased hyperparameter sensitivity — the balance between α, β, and temperature significantly affects convergence quality and requires careful tuning. Feature-level distillation has not yet been implemented and represents the primary remaining opportunity for improvement.

## ORIENT

### Strengths
- DKD substantially reduces time-to-target vs Vanilla KD (337.81s vs 493.6s)
- Separation of TCKD and NCKD gives finer control over learning dynamics
- Non-target supervision contributes strongly to convergence speed — dark knowledge among incorrect classes is highly informative for the ViT student
- Smoother optimization behavior and improved stability at higher learning rates

### Areas for Improvement
- Hyperparameter sensitivity is high — α, β, and temperature all require careful tuning
- Feature-level supervision not yet implemented — logit-only distillation may leave representation-level signal untapped
- Ablation studies isolating TCKD vs NCKD contributions have not been completed
- Multi-seed variance remains unquantified

### Critical Risks/Assumptions
- DKD gains assume carefully tuned hyperparameters — poorly tuned settings may underperform Vanilla KD
- Results are specific to ResNet-56 teacher and custom ViT student — generalization to other architectures unverified
- Single dataset (CIFAR-10) limits conclusions about broader applicability

## DECIDE

### Concrete Next Actions
- Implement feature-level distillation aligning ResNet-56 penultimate features with ViT CLS token representations
- Select appropriate layers for feature alignment and introduce learnable projection head
- Measure additional convergence gains from feature matching vs DKD alone
- Evaluate computational overhead introduced by feature alignment loss
- Continue refining α, β, and temperature through systematic sweeps

## ACT

### Resource Needs
- GPU compute time for feature distillation experiments and hyperparameter sweeps
- Reference material on feature alignment strategies for cross-architecture distillation
- Profiling tools to measure computational overhead of combined DKD and feature matching losses
- Structured experiment tracking to compare all four methods (baseline, Vanilla KD, DKD, Feature + DKD) under identical conditions
