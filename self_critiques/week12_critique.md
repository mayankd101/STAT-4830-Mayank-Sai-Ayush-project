# Self-Critique: Week 12 Decoupled Knowledge Distillation

## OBSERVE

After reviewing the Week 12 implementation and analyzing the most recent DKD experiments, the project has successfully transitioned from primarily systems-level optimization toward optimization-focused representation learning. The Decoupled Knowledge Distillation framework integrates correctly with the existing Vision Transformer training pipeline and produces a substantial reduction in wall-clock time-to-target accuracy compared to both the baseline ViT and Vanilla Knowledge Distillation configurations.

The separation between Target-Class Knowledge Distillation (TCKD) and Non-Target-Class Knowledge Distillation (NCKD) appears to significantly improve the quality of the optimization signal received by the student model. Training curves demonstrate noticeably faster early-stage convergence and reduced instability throughout training. The ViT student reaches the target accuracy threshold substantially earlier than previous approaches while maintaining stable optimization behavior under BF16 mixed precision and cosine scheduling.

The project now more clearly demonstrates that convergence efficiency and computational throughput are distinct optimization problems. Earlier systems-level optimizations improved hardware utilization and reduced per-step cost, but DKD directly reduces the number of optimization steps required to achieve the target accuracy threshold.

However, several limitations remain. The DKD objective introduces substantially higher hyperparameter sensitivity than Vanilla KD, particularly with respect to the balance between TCKD and NCKD weighting. Additionally, the current implementation still relies exclusively on output-level supervision. Intermediate representation alignment between the teacher and student networks has not yet been explored and may provide additional convergence improvements beyond logit-level distillation alone.

---

## ORIENT

### Strengths

- DKD produces a significant reduction in wall-clock time-to-target accuracy relative to both baseline ViT training and Vanilla KD.
- Separating TCKD and NCKD allows finer control over optimization dynamics and improves the quality of supervision provided to the student.
- Non-target class relationships (“dark knowledge”) appear to contribute strongly to early representation learning and convergence speed.
- The DKD pipeline integrates successfully with the existing BF16 mixed precision, cosine scheduling, and large-batch optimization framework.
- Validation curves are smoother and more stable than earlier distillation configurations, particularly during early-stage optimization.
- The project now has a substantially stronger research direction centered around convergence optimization rather than purely systems-level acceleration.

### Areas for Improvement

- Hyperparameter sensitivity remains high, particularly for DKD weighting parameters $\alpha$, $\beta$, and temperature $T$.
- The relative contributions of TCKD and NCKD have not yet been isolated through controlled ablation experiments.
- Feature-level supervision has not yet been implemented, potentially limiting the amount of transferable representation information.
- Multi-seed reproducibility experiments have not been conducted, making variance across runs unclear.
- Current experiments remain limited to CIFAR-10 and a single ViT configuration.

### Critical Risks / Assumptions

- The observed convergence gains assume carefully tuned DKD hyperparameters and may not generalize under different optimization settings.
- Improvements depend heavily on the quality and calibration of the ResNet-56 teacher model.
- Output-level supervision alone may eventually saturate, limiting further improvements in convergence efficiency.
- Results obtained on CIFAR-10 may not directly transfer to larger-scale datasets or deeper Vision Transformer architectures.
- Increased complexity in the DKD objective introduces additional implementation and debugging challenges.

---

## DECIDE

### Concrete Next Actions

- Implement feature-level distillation by aligning ResNet-56 penultimate feature representations with Vision Transformer CLS-token embeddings.
- Introduce a learnable projection layer to resolve dimensionality mismatches between CNN and transformer feature spaces.
- Conduct controlled ablation studies separating the effects of TCKD and NCKD individually.
- Perform systematic sweeps over DKD hyperparameters ($\alpha$, $\beta$, and temperature $T$) to identify more stable optimization regions.
- Evaluate the computational overhead introduced by feature-level supervision relative to its convergence benefits.
- Run multi-seed experiments to quantify variance in time-to-target accuracy and optimization stability.

---

## ACT

### Resource Needs

Further experimentation requires additional GPU compute time for repeated DKD and feature-distillation trials across multiple hyperparameter configurations and random seeds. More detailed profiling tools will also be useful for measuring the computational overhead introduced by combined logit-level and feature-level supervision.

Additional literature review on cross-architecture feature distillation and representation alignment strategies would strengthen the theoretical framing of the next stage of the project. More structured experiment tracking and configuration management are also needed to organize comparisons between baseline ViT training, Vanilla KD, DKD, and future feature-distillation approaches under identical training conditions.
