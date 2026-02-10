# Self-Critique: Week 4 ViT Training Pipeline

## OBSERVE

After re-reading the report and re-running the notebook, the training pipeline is functional and reproducible. The use of a pre-trained Vision Transformer allows the model to reach high accuracy in significantly fewer epochs, and logging for accuracy, throughput, and memory usage is consistent across runs. The training behavior is consistent with expectations, with decreasing loss and correctly shaped outputs for CIFAR-10 classification. However, early stopping is only partially implemented, augmentation strategies are limited, and several architectural and training hyperparameters remain unexplored.

## ORIENT

### Strengths
- Reproducible training pipeline with systematic logging of performance and resource metrics.
- Pre-trained Vision Transformer significantly reduces wall-clock time to reach target accuracy.
- Validation checks confirm correctness of loss computation and model outputs.

### Areas for Improvement
- Early stopping does not fully capture the exact time-to-target accuracy and needs refinement.
- Data augmentation could be expanded to improve generalization and convergence.
- Hyperparameters such as patch size, depth, and batch size have not been systematically evaluated.

### Critical Risks/Assumptions
- Results assume ImageNet pre-trained features transfer effectively to CIFAR-10 after resizing and normalization.
- Timing and throughput measurements assume consistent data-loading and batching behavior across runs.
- Accuracy and convergence trends assume the chosen patch size, depth, and optimizer settings are appropriate for CIFAR-10.


## DECIDE

### Concrete Next Actions
- Implement full early stopping to measure precise wall-clock time to 94% accuracy.
- Add and evaluate stronger augmentation strategies such as Mixup or RandAugment.
- Perform controlled hyperparameter sweeps for patch size, depth, learning rate, and batch size.

## ACT

### Resource Needs

- Deeper understanding of PyTorch AMP and gradient scaling to safely enable mixed precision training, using official PyTorch AMP documentation and tutorials.
- Time to run repeated controlled experiments to tune learning rate, batch size, and early stopping behavior while tracking time-to-target accuracy.
- Structured configuration management using YAML or JSON files to standardize experiments and reduce manual errors across runs.

