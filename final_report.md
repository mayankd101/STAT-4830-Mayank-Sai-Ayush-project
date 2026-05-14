# Reducing Time-to-Target Accuracy in Vision Transformers through Systems Optimization and Knowledge Distillation


## Ayush Tripathi, Sai Minnal, Mayank Deoras


---


# Abstract


Vision Transformers (ViTs) have emerged as a powerful alternative to convolutional neural networks for computer vision tasks due to their ability to model long-range dependencies through self-attention and scale effectively with larger datasets and model sizes. Their transformer-based architecture has enabled strong performance across image classification, segmentation, and multimodal learning systems, making them increasingly important in modern deep learning research and industry applications. 


Despite these advantages, ViTs are often computationally expensive and can converge slowly on smaller datasets compared to convolutional approaches. Most deep learning research focuses primarily on maximizing final predictive accuracy, while less attention is given to training efficiency and wall-clock convergence time. In practical machine learning systems, however, computational efficiency, hardware utilization, and iteration speed are critical constraints.


This project investigates methods for reducing the wall-clock time required for a Vision Transformer to achieve target accuracy on CIFAR-10. Rather than optimizing exclusively for final test accuracy, the project frames training as a time-to-target optimization problem. The work evolves through multiple stages: establishing a reproducible ViT benchmarking pipeline, implementing systems-level optimizations, analyzing the limitations of throughput-focused improvements, and ultimately pivoting toward knowledge distillation and representation-level supervision.


Initial experiments focused on systems optimizations including automatic mixed precision (AMP), BF16 computation, cosine learning-rate scheduling, data loading optimization, gradient clipping, Mixup augmentation, torch.compile, and a custom Muon optimizer. Although these techniques improved throughput and memory efficiency, they did not substantially reduce convergence time at higher accuracy thresholds.


The project then shifted toward improving learning efficiency through teacher–student distillation. Multiple distillation strategies were evaluated, including Vanilla Knowledge Distillation (KD), Decoupled Knowledge Distillation (DKD), and feature-level distillation combined with DKD. Experimental results demonstrate that feature-level supervision significantly accelerates convergence relative to baseline Vision Transformer training.


The primary conclusion of this project is that reducing training time requires improvements not only in computational throughput but also in optimization dynamics and representation learning. Systems-level optimizations reduce the cost of individual training steps, while distillation methods reduce the number of optimization steps required to achieve target accuracy.


---


# 1. Introduction


Deep learning systems are increasingly constrained not only by predictive performance but also by computational efficiency. Modern machine learning workflows require rapid experimentation, efficient hardware utilization, and reduced iteration cost. As a result, optimizing wall-clock training time has become an important systems and optimization problem.


Vision Transformers (ViTs) have emerged as a powerful architecture for image classification and representation learning. Unlike convolutional neural networks (CNNs), which encode strong spatial inductive biases through local receptive fields and weight sharing, Vision Transformers process images as sequences of patches using self-attention mechanisms. This flexibility allows transformers to scale effectively to large datasets and tasks, but it also introduces significant computational overhead and slower convergence behavior, particularly on smaller datasets such as CIFAR-10.


Many existing machine learning benchmarks emphasize final accuracy as the primary evaluation metric. However, final accuracy alone does not capture the practical efficiency of a model. Two models may achieve similar accuracy while requiring dramatically different amounts of training time and compute resources.


This project instead focuses on minimizing the wall-clock time required for a Vision Transformer to achieve a target test accuracy threshold on CIFAR-10. The project evolved substantially throughout development. Initial work focused on improving throughput and hardware efficiency through systems-level optimizations such as mixed precision, optimized data loading, learning-rate scheduling, and compiler-level acceleration. While these techniques improved per-step efficiency, they did not sufficiently address convergence inefficiency.


This observation motivated a shift toward knowledge distillation and optimization-focused learning strategies. The project ultimately investigated how teacher-guided supervision and representation alignment could improve learning efficiency and reduce time-to-target performance.


The final system combines:


* systems-level optimization
* output-level knowledge distillation
* feature-level representation supervision
* transformer-specific optimization techniques


into a unified Vision Transformer training pipeline.


The primary contribution of this work is demonstrating that convergence optimization and throughput optimization are fundamentally distinct problems. While systems optimizations reduce the computational cost of each iteration, distillation methods improve the quality of the optimization signal itself, allowing the model to reach target performance substantially faster.


---


# 2. Problem Formulation


## 2.1 Task Definition


The project focuses on CIFAR-10 image classification. CIFAR-10 contains:


* 50,000 training images
* 10,000 test images
* 10 semantic classes
* RGB images of size 32 × 32


The objective is not merely to maximize final test accuracy, but instead to minimize the elapsed wall-clock time required for the model to first achieve a target test accuracy threshold.


---


## 2.2 Time-to-Target Objective


Let:


* $A_{test}(t; \theta)$ denote test accuracy at wall-clock time $t$
* $\theta$ represent model architecture and training hyperparameters
* $\tau$ represent the target accuracy threshold


The optimization objective is:


$$
T_\tau(\theta)=\min { t \ge 0 : A_{test}(t;\theta)\ge\tau }
$$


The goal is therefore:


$$
\theta^* = \arg\min_\theta T_\tau(\theta)
$$


Unlike traditional training objectives, which optimize only final predictive accuracy, this formulation explicitly incorporates training efficiency into the optimization problem.


---


## 2.3 Vision Transformer Architecture


The Vision Transformer divides an image into non-overlapping patches.


If:


* image size = $H \times W$
* patch size = $P \times P$


then the total number of patches is:


$$
N = \left(\frac{H}{P}\right)\left(\frac{W}{P}\right)
$$


Each patch is flattened and projected into an embedding space of dimension $d$.


Positional embeddings are added:


$$
Z_0 = [z_{cls}; z_1; ... ; z_N] + E_{pos}
$$


Self-attention is computed as:


$$
\text{Attn}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


The attention matrix $QK^T$ scales quadratically with the number of patches:


$$
O(N^2)
$$


As a result, patch count and embedding dimensionality strongly influence computational efficiency.


---


# 3. Baseline Vision Transformer Pipeline


## 3.1 Initial Goal


The initial stage of the project focused on constructing a clean, reproducible Vision Transformer benchmarking pipeline for CIFAR-10.


The primary goals were:


* reproducible experiments
* trustworthy wall-clock measurements
* accurate throughput profiling
* scalable experimentation infrastructure


The baseline system established the experimental foundation used throughout the rest of the project.


---


## 3.2 Dataset Pipeline


The CIFAR-10 dataset was loaded using optimized PyTorch DataLoaders.


Standard augmentations included:


* random cropping
* horizontal flipping
* ImageNet normalization for pretrained models


The project supported both:


* pretrained ViTs (ImageNet initialization)
* ViTs trained from scratch


For pretrained Vision Transformers, CIFAR-10 images were resized to 224 × 224 to match ImageNet ViT input resolution.


---


## 3.3 Baseline ViT Configuration


The baseline Vision Transformer used:


* patch size = 4
* embedding dimension = 96
* depth = 4 transformer blocks
* 4 attention heads
* AdamW optimizer


Additional metrics logged included:


* throughput (images/sec)
* peak GPU memory
* wall-clock training time
* accuracy progression over epochs


The baseline training loop also implemented early stopping once the target accuracy threshold was reached.


---


## 3.4 Initial Findings


The baseline pipeline successfully established a reproducible benchmarking environment and stable training procedure.


However, several limitations became apparent:


* Vision Transformers converged relatively slowly on CIFAR-10
* throughput was constrained by hardware utilization
* training efficiency degraded significantly near higher accuracies
* final epochs exhibited diminishing returns


These observations motivated the next phase of the project: systems-level optimization.


---


# 4. Systems-Level Optimization


## 4.1 Motivation


The first major optimization phase focused on improving computational efficiency and hardware utilization.


The hypothesis was that reducing per-step training cost would substantially improve time-to-target performance.


This phase introduced multiple systems-level optimizations:


* AMP / BF16 mixed precision
* torch.compile
* cosine learning-rate scheduling
* warmup scheduling
* optimized data loading

---


## 4.2 Automatic Mixed Precision (AMP)


Automatic Mixed Precision reduces computational cost by using lower-precision arithmetic when numerically safe.


Two precision formats were investigated:


* FP16
* BF16


BF16 proved particularly effective on A100 GPUs because it provided:


* throughput improvements comparable to FP16
* improved numerical stability
* reduced overflow sensitivity


Mixed precision significantly improved:


* GPU utilization
* memory efficiency
* training throughput


while preserving stable optimization.


---


## 4.3 Learning Rate Scheduling


The project implemented cosine learning-rate scheduling with warmup.


Warmup scheduling gradually increases the learning rate during early epochs:


$$
\eta_t = \eta_{max}\frac{t}{T_{warmup}}
$$


Cosine annealing then reduces the learning rate smoothly:


$$
\eta_t = \frac{1}{2}\eta_{max}(1+\cos(\pi t/T))
$$


This scheduling strategy:


* improved optimization stability
* reduced early training divergence
* enabled larger learning rates
* accelerated convergence


---


## 4.4 Data Pipeline Optimization


The project optimized the data-loading pipeline using:


* persistent workers
* prefetch_factor
* optimized augmentation pipelines
* non-deterministic CUDA benchmarking


These changes reduced CPU-side bottlenecks and improved GPU utilization.


---

## 4.5 Results of Systems Optimization


Systems-level optimizations successfully improved:


* throughput
* memory usage
* GPU efficiency
* per-step training cost


Representative results included:


| Metric          | Baseline   | Optimized  |
| --------------- | ---------- | ---------- |
| Best Accuracy   | 94.1%      | 94.2%      |
| Time-to-Target  | 804s       | 668s       |
| Throughput      | 2850 img/s | 3450 img/s |
| Peak GPU Memory | 4.2 GB     | 3.1 GB     |


Despite these improvements, convergence near higher accuracy thresholds remained inefficient.


This led to the central insight of the project:


> improving throughput alone does not sufficiently optimize convergence behavior.


The project therefore pivoted toward improving learning efficiency directly.


---


# 5. Knowledge Distillation


## 5.1 Motivation for Distillation


The project observed that convolutional neural networks converged substantially faster than Vision Transformers on CIFAR-10.


CNNs possess strong inductive biases:


* locality
* translation invariance
* hierarchical feature extraction


which improve sample efficiency on small datasets.


The project hypothesized that teacher-guided supervision could transfer these advantages to a Vision Transformer student while preserving transformer flexibility.


This motivated the transition toward knowledge distillation.


---


## 5.2 Vanilla Knowledge Distillation


The first distillation method implemented was Vanilla Knowledge Distillation (KD).


A pretrained ResNet-56 model served as the teacher network, while the Vision Transformer acted as the student.


The training objective combined:


* hard-label cross entropy
* KL divergence between teacher and student logits


The overall loss was:


$$
L = (1-\alpha)L_{CE} + \alpha L_{KD}
$$


where:


$$
L_{KD} = KL\left(\sigma(z_s/T), \sigma(z_t/T)\right)
$$


and:


* $z_s$ = student logits
* $z_t$ = teacher logits
* $T$ = distillation temperature


Teacher soft targets encode inter-class relationships often referred to as “dark knowledge.”


---


## 5.3 Observations from Vanilla KD


Vanilla KD improved:


* convergence smoothness
* optimization stability
* early learning behavior


However, improvements in time-to-target were relatively modest.


One limitation of Vanilla KD is that it treats all non-target classes similarly, preventing finer control over the optimization signal.


This motivated the investigation of more advanced distillation methods.


---


# 6. Teacher Architecture Experiments


## 6.1 Motivation


An important question emerged during experimentation:


> Is the benefit of distillation primarily caused by CNN inductive bias, or simply by soft-label regularization?


To investigate this, the project evaluated multiple teacher configurations.


---


## 6.2 Experimental Conditions


Four conditions were tested:


| Condition             | Description                       |
| --------------------- | --------------------------------- |
| Label Smoothing       | No teacher, CE only               |
| EMA Self-Distillation | Student distills from EMA weights |
| ViT Teacher           | Distillation from another ViT     |
| CNN Teacher           | Distillation from ResNet-56       |


All conditions used:


* identical student architectures
* identical optimizers
* identical schedules
* identical augmentation pipelines


---


## 6.3 Findings


Results indicated that CNN teachers consistently produced stronger convergence improvements than ViT teachers or EMA self-distillation.


This suggested that:


* CNN inductive bias plays a major role in convergence acceleration
* distillation benefits extend beyond simple soft-label regularization
* representation structure from CNN teachers provides valuable optimization guidance


These findings further motivated the project’s focus on teacher-guided learning.


---


# 7. Decoupled Knowledge Distillation (DKD)


## 7.1 Motivation


Vanilla KD uses a single KL divergence term over the entire teacher softmax distribution.


However, this combines:


* target-class alignment
* non-target relational structure


into one objective.


The project therefore implemented Decoupled Knowledge Distillation (DKD), which separates these signals.


---


## 7.2 DKD Formulation


DKD decomposes distillation into:


* TCKD (Target Class Knowledge Distillation)
* NCKD (Non-Target Class Knowledge Distillation)


The total distillation loss becomes:


$$
L_{DKD} = \alpha L_{TCKD} + \beta L_{NCKD}
$$


where:


* $\alpha$ controls target-class emphasis
* $\beta$ controls non-target class relationships


TCKD models:


* probability mass on the correct class
* probability mass on all incorrect classes


NCKD masks the target class and focuses exclusively on dark knowledge among distractor classes.


---


## 7.3 Implementation Details


The DKD implementation followed the Megvii mdistiller reference implementation.


The project additionally introduced:


* DKD warmup scheduling
* BF16 training
* cosine scheduling
* large-batch optimization
* sparse validation strategies


The ViT student architecture used:


* embedding dimension = 192
* 6 transformer blocks
* 6 attention heads


while the teacher remained a pretrained ResNet-56.


---


## 7.4 DKD Results


DKD substantially improved convergence behavior relative to Vanilla KD.


Key improvements included:


* faster early-stage convergence
* smoother optimization dynamics
* stronger representation learning
* improved time-to-target performance


Experimental results showed that preserving non-target class relationships was particularly important for accelerating learning.


The project concluded that dark knowledge among incorrect classes provides a strong optimization signal for transformer training.


---


# 8. Feature Distillation


## 8.1 Motivation


Although DKD improved convergence substantially, it still relied primarily on output-level supervision.


The project hypothesized that matching intermediate feature representations could provide stronger and denser supervision signals.


Instead of supervising only logits, feature distillation aligns internal representations between teacher and student networks.


---


## 8.2 Feature Alignment Strategy


The teacher feature was extracted from:


* the ResNet-56 global average pooling layer


The student feature was extracted from:


* the Vision Transformer CLS token representation before classification.


Because feature dimensionalities differed, the project introduced a learnable linear projection:


$$
\mathbb{R}^{d_s} \rightarrow \mathbb{R}^{d_t}
$$


aligning student and teacher feature spaces.


---


## 8.3 Feature Distillation Loss


Feature alignment was modeled using mean squared error:


$$
L_{feature} = ||f_s - f_t||_2^2
$$


The final optimization objective became:


$$
L = L_{CE} + \alpha L_{DKD} + \gamma L_{feature}
$$


where:


* $f_s$ = projected student features
* $f_t$ = teacher features
* $\gamma$ controls feature alignment strength


The project additionally explored cosine-similarity feature losses.


---


## 8.4 Why Feature Distillation Helps


Logit-based supervision provides only sparse information over class probabilities.


Feature matching instead provides:


* dense representation-level supervision
* per-sample alignment signals
* earlier optimization guidance


This allows the student model to learn meaningful representations substantially earlier in training.


---


## 8.5 Results


Feature distillation combined with DKD produced the strongest performance observed during the project.


The combined system:


* converged faster
* produced smoother optimization curves
* reduced instability
* achieved the best time-to-target results


Feature alignment loss decreased consistently throughout training, indicating successful representation transfer.


The project concluded that feature-level supervision improves optimization efficiency more effectively than output-level supervision alone.


---


# 9. Experimental Results


## 9.1 Evaluation Metrics


The project evaluated models using:


* time-to-target accuracy
* throughput (images/sec)
* wall-clock training time
* peak GPU memory usage
* convergence stability


The primary metric throughout the project was wall-clock time required to reach the target accuracy threshold.


---


## 9.2 Overall Comparison


| Method               | Time-to-Target       | Key Insight                |
| -------------------- | -------------------- | -------------------------- |
| Baseline ViT         | Slow convergence     | No optimization            |
| Systems Optimization | Faster throughput    | Better per-step efficiency |
| Vanilla KD           | Smoother convergence | Teacher soft labels        |
| DKD                  | Major improvement    | Better optimization signal |
| Feature + DKD        | Best overall         | Representation supervision |


---


## 9.3 Systems Optimization vs Distillation


One of the most important conclusions from the project is that systems optimization and distillation improve fundamentally different aspects of training.


Systems optimization improves:


* throughput
* memory efficiency
* hardware utilization
* computational cost per iteration


Distillation improves:


* optimization dynamics
* representation learning
* convergence speed
* learning efficiency


The project found that reducing the number of optimization steps required was ultimately more important than reducing the cost of each individual step.


---


# 10. Discussion


The project evolved substantially over time, ultimately revealing several important insights about efficient transformer training.


The initial assumption was that computational bottlenecks were the primary limiting factor for Vision Transformer efficiency. This motivated the systems optimization phase.


However, empirical results demonstrated that even substantial throughput improvements failed to fully address convergence inefficiency.


This led to the project’s central conceptual insight:


> throughput optimization and convergence optimization are fundamentally distinct problems.


Systems optimizations reduced the cost of each iteration, but distillation methods reduced the number of iterations required.


The strongest improvements ultimately came from:


* better supervision signals
* representation alignment
* improved optimization dynamics


rather than purely hardware-focused improvements.


The project also demonstrated that:


* CNN inductive bias transfers effectively through distillation
* feature supervision improves representation learning
* non-target class structure is important for transformer optimization


Overall, the project increasingly shifted from a systems-engineering perspective toward a representation-learning and optimization perspective.


---


# 11. Limitations and Future Work


## 11.1 Limitations


Several limitations remain:


* experiments were limited to CIFAR-10
* only a single ViT scale was tested extensively
* feature alignment introduced hyperparameter complexity
* DKD required substantial tuning
* no ImageNet-scale experiments were performed


Additionally, some experiments prioritized training efficiency over absolute final accuracy.


---


## 11.2 Future Work


Future directions include:


* scaling experiments to ImageNet
* evaluating larger Vision Transformers
* multi-teacher distillation
* self-supervised teachers
* DeiT-style training strategies
* pretrained initialization methods
* representation-learning analysis
* distillation for multimodal transformers


Additional work could also investigate:


* sparse attention architectures
* token pruning
* adaptive computation
* low-rank attention methods


as complementary approaches to reducing transformer training cost.


---


# 12. Conclusion


This project investigated methods for reducing the wall-clock time required for Vision Transformers to achieve target accuracy on CIFAR-10.


The work evolved through multiple stages:


1. reproducible benchmarking infrastructure
2. systems-level optimization
3. throughput-focused acceleration
4. knowledge distillation
5. feature-level representation supervision


Systems-level optimizations successfully improved:


* throughput
* memory efficiency
* hardware utilization


However, these techniques alone were insufficient to fully optimize convergence behavior.


The project ultimately demonstrated that improving learning efficiency through distillation produced substantially larger gains in time-to-target performance.


Among all approaches explored, feature distillation combined with Decoupled Knowledge Distillation produced the strongest convergence improvements.


The central conclusion of the project is that efficient deep learning depends not only on faster computation, but also on better optimization signals and representation learning.


In modern machine learning systems, reducing training time requires improving both:


* computational efficiency
* convergence efficiency


simultaneously.


---


# References


1. Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*


2. Hinton et al., *Distilling the Knowledge in a Neural Network*


3. Zhao et al., *Decoupled Knowledge Distillation*


4. Keller Jordan, *modded-nanoGPT*


5. Keller Jordan, *cifar10-airbench*


6. PyTorch Documentation


7. chenyaofo/pytorch-cifar-models



