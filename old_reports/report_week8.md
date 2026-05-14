# Week 8: Vision Transformer Wall-Clock Optimization for CIFAR-10

## Problem Statement

The objective of this project is to minimize the wall-clock time required for a Vision Transformer (ViT) to achieve **94% test accuracy** on the CIFAR-10 image classification benchmark. Rather than optimizing exclusively for final predictive accuracy, this work emphasizes **time-to-target performance**, defined as the elapsed training time required for the model to first satisfy a specified accuracy threshold.

This problem is significant because modern deep learning systems are increasingly constrained not only by statistical performance but also by computational efficiency, hardware utilization, and iteration speed. In practical machine learning workflows, faster convergence directly reduces experimentation cost and improves scalability. Vision Transformers have demonstrated strong performance on image classification tasks, but they are often computationally expensive and may converge more slowly than convolutional architectures when applied to smaller datasets. Consequently, identifying optimization strategies that reduce convergence time while maintaining accuracy is an important systems and learning problem.

The CIFAR-10 dataset is used as the experimental benchmark. It consists of 60,000 RGB images of size $32 \times 32$, distributed across 10 semantic classes, with 50,000 training images and 10,000 test images. The primary success metric is the **time required to achieve 94% test accuracy**. Secondary evaluation metrics include training throughput measured in images per second, total runtime, peak GPU memory usage, and the stability of the training and validation accuracy curves.

The main constraints include GPU memory limitations, numerical stability under aggressive optimization settings, and sensitivity of transformer architectures to hyperparameter selection. The project requires access to CIFAR-10 image data, training logs, runtime profiling metrics, and hardware utilization measurements. Potential failure modes include failure to reach the target accuracy threshold, instability caused by large learning rates or reduced precision, and data-loading bottlenecks that reduce effective GPU utilization.

---

## Technical Approach

The optimization problem is formulated as a constrained time-to-target learning objective. The primary goal is to minimize the total wall-clock training time required for a Vision Transformer to achieve at least 94% test accuracy on CIFAR-10. Formally, this objective is written as

$$
\min T
$$

subject to

$$
A_{\text{test}} \geq 94\%
$$

where $T$ denotes elapsed training time and $A_{\text{test}}$ represents test-set classification accuracy.

At the statistical learning level, the model parameters are optimized using the standard cross-entropy loss function

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{i,c}\log(\hat{p}_{i,c}),
$$

where $N$ is the batch size, $C=10$ is the number of classes, $y_{i,c}$ is the one-hot encoded ground-truth label, and $\hat{p}_{i,c}$ is the predicted probability for class $c$.

The underlying model is a custom Vision Transformer adapted specifically for CIFAR-10. Each image is partitioned into non-overlapping patches of size $4 \times 4$, which are linearly projected into an embedding space of dimension 192. These embeddings are processed through a stack of six transformer encoder blocks, each with six attention heads. Instead of using a dedicated classification token, the architecture employs global average pooling over patch representations:

$$
z = \frac{1}{N_p}\sum_{i=1}^{N_p} h_i,
$$

where $N_p$ is the number of patches and $h_i$ denotes the latent representation of the $i$-th patch. This design reduces parameter overhead and improves convergence stability on smaller datasets.

To improve computational efficiency, the PyTorch implementation integrates several systems-level optimizations. First, BF16 automatic mixed precision is used to reduce memory consumption and accelerate matrix operations while maintaining numerical stability. This significantly improves throughput during both forward and backward passes.

Second, the implementation uses `torch.compile`, which performs graph-level optimization and kernel fusion. This reduces Python execution overhead and improves GPU utilization by combining tensor operations into optimized kernels.

The optimization algorithm combines a custom Muon optimizer for transformer weight matrices with AdamW updates for one-dimensional parameters such as biases and normalization terms. The update rule is given by

$$
W_{t+1} = W_t - \eta \hat{G}_t,
$$

where $\eta$ is the learning rate and $\hat{G}_t$ denotes the orthogonalized gradient direction obtained through Newton–Schulz-based matrix orthogonalization.

To improve generalization and stabilize training, the pipeline incorporates label smoothing and Mixup augmentation. Label smoothing modifies the target distribution as

$$
y' = (1-\epsilon)y + \frac{\epsilon}{K},
$$

where $\epsilon = 0.1$ and $K=10$.

Validation is performed after each epoch using the held-out test set. In addition to accuracy, the system records throughput, peak GPU memory usage, and exact time-to-target measurements. Resource requirements include GPU acceleration, sufficient memory for transformer training, and runtime profiling tools.

---

## Initial Results

The current implementation executes successfully end-to-end and demonstrates stable training behavior. Forward and backward propagation complete without numerical instability, and all optimization components integrate correctly into the PyTorch training pipeline.

Initial experiments indicate improved computational efficiency relative to baseline transformer training configurations. BF16 mixed precision substantially reduces memory usage and improves throughput, allowing more efficient hardware utilization. The use of `torch.compile` further reduces per-iteration overhead, particularly during repeated attention and feedforward operations.

Preliminary validation results show consistent improvement in test accuracy across epochs, with smoother convergence under the optimized training setup. Throughput measurements indicate a meaningful increase in images processed per second compared to standard FP32 training. Peak GPU memory consumption is reduced sufficiently to support larger effective batch sizes.

Current limitations include incomplete hyperparameter tuning, sensitivity to optimizer settings, and uncertainty regarding the relative contribution of each optimization technique. Additional profiling is required to isolate the performance impact of mixed precision, compiler optimization, and custom optimizer geometry.

Unexpected challenges include instability during early training when aggressive learning rates are used, as well as occasional data-loading bottlenecks that reduce effective GPU utilization.

---

## Next Steps

The immediate next step is to perform controlled ablation studies to quantify the contribution of each optimization component individually. In particular, the effects of BF16 precision, `torch.compile`, the Muon optimizer, Mixup augmentation, and label smoothing must be isolated through systematic experimentation.

A key technical challenge is determining the optimal trade-off between aggressive optimization settings and training stability. Higher learning rates and reduced precision can accelerate convergence but may also introduce gradient instability.

Further work is needed to refine the optimizer configuration, improve scheduler design, and tune batch size for maximum throughput. Questions that require further investigation include whether deeper transformer architectures benefit similarly from these optimizations and whether alternative optimizers such as AdamW-only baselines can achieve comparable time-to-target performance.

Alternative approaches include comparing this transformer pipeline against strong convolutional baselines and exploring pretrained initialization strategies.

The most important lesson learned so far is that model architecture alone does not determine efficiency. Systems-level considerations such as numerical precision, compilation strategy, and optimization geometry play a central role in reducing wall-clock convergence time.
