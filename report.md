# Week 4: Vision Transformer (ViT) Training Pipeline for CIFAR-10

## Problem Statement

The aim of this project is to develop a fully reproducible training pipeline for a Vision Transformer (ViT) applied to the CIFAR-10 dataset. Our primary goal is to minimize the wall-clock time required for the model to achieve a target test accuracy of 94%. Efficient training pipelines are essential for research and experimentation, as they allow rapid testing of different model architectures, hyperparameters, and optimization strategies. CIFAR-10 is a widely used benchmark dataset consisting of 50,000 training images and 10,000 test images across 10 classes, providing a controlled environment for evaluating both pre-trained and scratch ViT models.  

We measure success primarily by the time it takes to reach 94% accuracy. Secondary considerations include throughput in images per second, peak GPU memory usage, and the stability of training and validation accuracy over time. Key constraints include reproducibility, ensured by fixing a random seed, and hardware limitations that affect batch size and model configuration. Pre-trained models require additional preprocessing such as resizing images to 224×224 pixels and normalizing them using ImageNet statistics. Potential challenges include failing to reach the target accuracy within the capped number of epochs, slowdowns caused by deterministic training modes, and limitations in GPU memory or compute resources. Addressing these challenges while maintaining reproducibility is crucial for meaningful benchmarking and for subsequent experiments in model optimization.

---

## Technical Approach

Our objective is to optimize the wall-clock **time to reach 94% test accuracy** on the CIFAR-10 dataset using a Vision Transformer (ViT) model. Formally, we minimize the cross-entropy loss over the training set:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log \hat{p}_{i,c}
$$

where \(y_{i,c}\) is the one-hot label for sample \(i\) and class \(c\), and \(\hat{p}_{i,c}\) is the predicted probability from the model. The problem is constrained by reproducibility requirements, limited GPU memory, and the CIFAR-10 image size (32×32 for scratch, 224×224 for pre-trained).

We employ a **Vision Transformer** because of its ability to capture global context through self-attention, which can improve convergence speed. For pre-trained experiments, we fine-tune ImageNet weights, replacing the classification head to accommodate the 10 CIFAR-10 classes. For training from scratch, we construct a ViT with 4 layers, 4 attention heads, and embedding dimension 96.

The PyTorch implementation strategy involves modular code: `model.py` for the ViT architecture, `utils.py` for data loading, training, evaluation, and device management. We use `train` and `validate` functions with early stopping when 94% accuracy is reached. To ensure reproducibility, we fix random seeds and log both throughput and peak GPU memory.

Validation is performed by checking the model output shapes, confirming finite loss values, and plotting train/test accuracy curves over epochs. Resource monitoring includes peak GPU memory usage and training throughput per batch. These measures allow us to track efficiency improvements systematically and ensure all modifications are reproducible.


---

## Initial Results

The implementation has been successfully executed, with preliminary results confirming that the pipeline functions as intended. A forward pass through the pre-trained ViT demonstrates that output shapes are correct, producing a tensor of size `(batch_size, 10)` for CIFAR-10. The pre-trained model contains approximately 5.5 million parameters, and the training loop correctly logs throughput, peak GPU memory, and epoch-wise accuracy metrics. Reproducibility has been verified by repeating runs with the same random seed, yielding consistent results.  

Basic test cases validate the implementation, confirming that outputs have the expected shape and that the loss is finite. Performance metrics from initial runs include batch throughput, wall-clock time to reach accuracy milestones, and GPU memory usage. Current limitations include the potential failure to reach the 94% accuracy target within the capped number of epochs, the simplicity of the augmentation strategy, and slowdown caused by deterministic settings. Despite these limitations, the pipeline is functional and provides reliable metrics that can guide further optimization and experimentation. These initial results form the foundation for iterative improvements and benchmarking of alternative approaches.

---

## Next Steps

Immediate improvements involve implementing full early stopping to accurately capture the exact wall-clock time to reach the target accuracy and performing hyperparameter tuning to ensure the model consistently achieves 94% accuracy. Additional enhancements include experimenting with more advanced data augmentations such as Mixup or RandAugment, which could improve generalization and convergence speed. Mixed precision training (AMP) will be explored to reduce memory usage and training time.  

Technical challenges to address include optimizing throughput and memory for larger batch sizes, integrating structured configuration files (YAML/JSON) to streamline experimentation, and exploring alternative ViT variants for speed and efficiency. Questions remain regarding the optimal patch size and depth for CIFAR-10, as well as the best augmentation strategies for faster convergence. Alternative approaches under consideration include comparing scratch versus pre-trained ViTs, implementing learning rate schedulers, and testing smaller or more efficient model architectures. Through this project, we have learned the importance of a modular repository structure, the effectiveness of pre-trained models in reducing training time, and the value of detailed resource monitoring for reproducible and meaningful benchmarking.


