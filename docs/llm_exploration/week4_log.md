# Week 4 LLM Exploration Summary

## Session Focus
These conversations were sparked by the need to design a reproducible and efficient Vision Transformer training pipeline for CIFAR-10. The focus was on identifying metrics, implementation strategies, and experiment logging practices that would improve model benchmarking and report clarity.

---

## Conversation 1: Time-to-Target as a Primary Metric

**Prompt That Worked:**  
- “How should we measure training efficiency beyond final accuracy for deep learning models?”

**Key Insights:**  
- Wall-clock time to reach a fixed accuracy is more informative than epoch count for comparing models.  
- Early stopping tied to accuracy enables direct comparison between different architectural or hyperparameter changes.  
- Logging throughput alongside time helps explain convergence speed differences even if learning curves appear similar.

**Techniques That Worked:**  
- Framing questions around efficiency and benchmarking produced metrics we could implement directly.  

**Dead Ends:**  
- Initially considering only final accuracy did not highlight the practical benefits of pre-trained models.  

**Next Steps:**  
- Implement time-to-target as the primary efficiency metric in the training loop.  

---

## Conversation 2: Pre-trained vs Scratch ViT on CIFAR-10

**Prompt That Worked:**  
- “What are the tradeoffs between training a ViT from scratch versus fine-tuning a pre-trained ViT on CIFAR-10?”

**Key Insights:**  
- Pre-trained ViTs reach high accuracy much faster and reduce wall-clock time drastically.  
- Using pre-trained models requires resizing to 224×224 and normalizing with ImageNet statistics, which adds preprocessing overhead.  
- Scratch ViTs allow testing architectural variations but need more tuning to achieve competitive accuracy.

**Techniques That Worked:**  
- Asking for tradeoffs produced actionable advice for choosing which model to benchmark first.  

**Dead Ends:**  
- Trying to adjust architecture and optimization at the same time without baseline comparison made iteration difficult.  

**Next Steps:**  
- Use a pre-trained ViT as the baseline and compare scratch models only after establishing consistent logging.

---

## Conversation 3: Reproducibility and Logging

**Prompt That Worked:**  
- “What should we log to make deep learning experiments reproducible and comparable?”

**Key Insights:**  
- Fixing random seeds is necessary but insufficient
- Deterministic transforms and consistent data loading are important  
- Logging accuracy, throughput, and peak memory per epoch enables analysis of efficiency trade-offs.  
- Returning histories from training simplifies plotting, validation, and benchmarking.

**Techniques That Worked:**  
- Asking for specific logging recommendations allowed us to implement modular metric tracking.  

**Dead Ends:**  
- Logging only final metrics made it impossible to verify reproducibility or compare intermediate performance.  

**Next Steps:**  
- Implement structured logging of epoch-level metrics and histories for both train and test phases.

---

## Conversation 4: Data Augmentation Strategies

**Prompt That Worked:**  
- “Which augmentations improve convergence speed and generalization for CIFAR-10 on ViT?”

**Key Insights:**  
- RandomCrop and RandomHorizontalFlip are sufficient for initial benchmarks.  
- Mixup or RandAugment could improve generalization and convergence speed in later experiments.  
- Overly complex augmentations initially slowed iteration without measurable gains.

**Techniques That Worked:**  
- Asking about minimal but effective augmentation strategies helped to guide the design of the baseline pipeline.  

**Dead Ends:**  
- Adding complex augmentations from the start was counterproductive.  

**Next Steps:**  
- Introduce Mixup or RandAugment in controlled experiments once baseline is established.

---

## Conversation 5: Hyperparameter Tuning and Early Stopping

**Prompt That Worked:**  
- “How should we tune hyperparameters and implement early stopping for a ViT on CIFAR-10?”

**Key Insights:**  
- Hyperparameter tuning is most effective after baseline results are established with a pre-trained model.  
- Early stopping should monitor test accuracy to capture exact wall-clock time to target.  
- Logging intermediate performance metrics prevents misinterpretation of transient fluctuations.

**Techniques That Worked:**  
- Focusing on specific goals (time-to-target) made early stopping implementation a lot more straightforward.  

**Dead Ends:**  
- Using early stopping without logging led to ambiguous measurements of convergence time.  

**Next Steps:**  
- Implement full early stopping and run systematic hyperparameter sweeps.

---

Note: This summary was [written by us | helped by ChatGPT | verified by Claude ] 
