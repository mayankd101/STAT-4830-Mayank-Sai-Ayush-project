# Week 6: Vision Transformer (ViT) Wall-Clock Optimization for CIFAR-10

## Problem Statement

The objective of Week 6 is to reduce the wall-clock time required for a Vision Transformer (ViT) to reach 94% test accuracy on the CIFAR-10 dataset by introducing targeted system-level optimizations to the existing training pipeline. While Week 4 focused on building a fully reproducible and modular implementation, this phase emphasizes computational efficiency, convergence speed, and hardware utilization. The central question is no longer whether the model can reach 94% accuracy, but how quickly it can do so under controlled and measurable conditions.

CIFAR-10 remains the benchmark dataset, consisting of 50,000 training images and 10,000 test images across 10 classes. The primary metric is the time required to reach 94% test accuracy. Secondary metrics include training throughput measured in images per second, peak GPU memory usage, total wall-clock training time, and the stability of training and validation accuracy curves. Constraints include GPU memory limitations that restrict batch size, preprocessing requirements for pre-trained models such as resizing to 224×224 with ImageNet normalization, and the trade-off between deterministic reproducibility and maximum execution speed. Potential challenges include failing to reach the target accuracy within the maximum epoch limit, instability introduced by aggressive optimization strategies, and data-loading bottlenecks that reduce effective hardware utilization. The goal is to accelerate convergence while maintaining reliable and interpretable benchmarking.

---

## Technical Approach

Our objective remains the minimization of cross-entropy loss over the training data:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log \hat{p}_{i,c}
$$

where $y_{i,c}$ is the one-hot encoded ground-truth label and $\hat{p}_{i,c}$ is the predicted class probability. However, in Week 6 the optimization target is reframed as minimizing wall-clock time subject to the constraint that test accuracy reaches or exceeds 94%. This transforms the problem from pure statistical optimization to a joint optimization over both learning dynamics and computational efficiency.

The primary architectural configuration uses a pre-trained ViT Tiny model initialized with ImageNet weights and adapted to CIFAR-10 classification. Transfer learning significantly accelerates convergence compared to training from scratch, allowing improvements in system efficiency to translate directly into reductions in time-to-target.

Several system-level optimizations were introduced. Automatic Mixed Precision (AMP) is enabled on CUDA devices, allowing FP16 computation during forward and backward passes while preserving numerical stability through gradient scaling. This increases throughput and reduces GPU memory usage. A cosine annealing learning rate scheduler with a three-epoch warmup phase is integrated to encourage faster early convergence and smoother late-stage optimization. A deterministic toggle is introduced to explicitly control the trade-off between strict reproducibility and execution speed, with nondeterministic mode enabling `cudnn.benchmark=True` for improved performance. The data-loading pipeline is optimized using persistent workers and prefetching to reduce batch latency and prevent GPU underutilization. Optional stronger augmentation such as RandAugment can be enabled to study its impact on convergence speed.

The implementation remains modular, with model definitions and utility functions separated from execution logic. The training loop integrates AMP support, scheduler updates, early stopping once 94% test accuracy is achieved, throughput logging, peak GPU memory tracking, and precise wall-clock time measurement. Validation includes verifying output tensor shapes of `(batch_size, 10)`, ensuring finite loss values, confirming consistent accuracy progression, and testing reproducibility under fixed random seeds. Controlled comparisons are performed between baseline configurations without AMP or scheduler support and optimized configurations with all enhancements enabled.

---

## Initial Results

The optimized pipeline executes successfully end-to-end and preserves all correctness guarantees established in Week 4. Forward passes produce correctly shaped outputs, and loss values remain finite throughout training. Accuracy curves demonstrate stable convergence behavior, with faster early-stage improvement when cosine scheduling and mixed precision are enabled.

Throughput increases noticeably under AMP, reflecting improved hardware utilization. Peak GPU memory usage decreases due to FP16 computation, allowing more efficient batch processing. When nondeterministic mode is enabled, total training time decreases further due to backend performance optimizations. Most importantly, the system is now capable of measuring precise time-to-target accuracy, enabling direct benchmarking of configuration choices.

While convergence speed improves under the optimized configuration, sensitivity to batch size, augmentation strength, and learning rate remains an important consideration. Deterministic execution still provides slightly more stable reproducibility but at the cost of increased training time. Overall, the results confirm that system-level optimizations meaningfully reduce wall-clock convergence time without sacrificing model correctness or stability.

---

## Next Steps

Future work will focus on quantifying the individual contribution of each optimization component through controlled ablation experiments. Measuring exact speedup factors for AMP, scheduler integration, and data-loader tuning will clarify which interventions produce the greatest efficiency gains. Hyperparameter sweeps over batch size, learning rate, and augmentation strength will help identify configurations that consistently reach 94% accuracy with minimal time-to-target. Additional experimentation with lightweight ViT variants may further improve computational efficiency.

Further technical improvements include refining early stopping to capture sub-epoch timing precision, integrating structured configuration management for systematic experimentation, and analyzing the trade-off between strict reproducibility and maximum hardware throughput. 

Week 6 demonstrates that training efficiency is not solely determined by model architecture or loss minimization, but also by numerical precision strategies, scheduler design, and data pipeline engineering. Optimizing deep learning systems therefore requires attention to both statistical learning principles and computational systems design.
