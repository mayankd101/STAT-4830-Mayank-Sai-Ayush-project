# Week 10 LLM Exploration Summary

## Session Focus
These conversations focused on **transitioning from system-level optimization to knowledge distillation** as the primary strategy for reducing wall-clock time required for a Vision Transformer to reach **85% CIFAR-10 test accuracy**. The discussions emphasized **teacher-student frameworks, soft target supervision, and distillation loss design**.

---

## Conversation 1: Teacher-Student Distillation Framework

**Prompt That Worked:**
"How does knowledge distillation improve convergence speed for Vision Transformers on small datasets?"

**Key Insights:**
CNN teachers converge faster on small datasets due to stronger inductive biases — locality and translation invariance. By transferring soft targets from a pretrained ResNet-56 teacher, the ViT student receives richer supervision than one-hot labels alone, improving optimization efficiency early in training.

**Techniques That Worked:**
Framing distillation as a **convergence acceleration strategy** rather than a compression technique produced more relevant implementation insights.

**Dead Ends:**
Treating distillation purely as a model compression tool missed its direct relevance to time-to-target optimization.

**Next Steps:**
Implement Vanilla KD with ResNet-56 teacher and measure exact time-to-target improvements relative to the optimized baseline.

---

## Conversation 2: Soft Targets and Temperature Scaling

**Prompt That Worked:**
"What is the role of temperature in knowledge distillation and how does it affect the learning signal?"

**Key Insights:**
Temperature scaling smooths the teacher's probability distribution, revealing inter-class similarity structure — often called dark knowledge. Higher temperatures produce softer distributions that expose more relational information between classes, while lower temperatures collapse back toward hard labels.

**Techniques That Worked:**
Asking specifically about **the information content of non-target class probabilities** clarified why soft targets improve convergence over one-hot labels.

**Dead Ends:**
Focusing only on the target class probability missed the key contribution of dark knowledge among non-target classes.

**Next Steps:**
Tune temperature T and weighting α systematically and measure their effect on time-to-target performance.

---

## Conversation 3: Vanilla KD Limitations

**Prompt That Worked:**
"Why does Vanilla KD treat all non-target classes equally and why is that a limitation?"

**Key Insights:**
Vanilla KD applies a single KL divergence term over the full teacher softmax, which entangles target-class alignment with non-target relational structure. This prevents independent control over how much the student learns from correctness versus inter-class relationships.

**Techniques That Worked:**
Asking about **the structure of the KL divergence term** specifically led directly to understanding the DKD motivation.

**Dead Ends:**
Evaluating Vanilla KD only by final accuracy missed its limitations in terms of optimization dynamics and convergence speed.

**Next Steps:**
Implement Decoupled Knowledge Distillation to separately control target-class and non-target supervision signals.

---

## Conversation 4: CNN vs ViT Teacher Comparison

**Prompt That Worked:**
"Does the benefit of distillation come from CNN inductive bias or just from soft-label regularization?"

**Key Insights:**
CNN teachers consistently outperform ViT teachers and EMA self-distillation in convergence acceleration on small datasets. This suggests that CNN inductive bias — not just soft-label regularization — plays a major role in transferring useful optimization signal to the student.

**Techniques That Worked:**
Designing a **controlled teacher comparison experiment** with identical student architectures produced cleaner attribution of distillation benefits.

**Dead Ends:**
Assuming all teacher types produce equivalent soft labels overlooked the role of representation structure in distillation quality.

**Next Steps:**
Use CNN teacher exclusively for remaining distillation experiments given its consistently stronger convergence improvements.

---

## Conversation 5: Integrating Distillation with System Optimizations

**Prompt That Worked:**
"How should we combine knowledge distillation with existing BF16 and cosine scheduling optimizations?"

**Key Insights:**
Distillation loss functions are compatible with AMP, cosine scheduling, and AdamW without modification. The combined system benefits from both faster per-step computation and richer learning signal simultaneously.

**Techniques That Worked:**
Verifying **numerical stability of KL divergence under BF16 precision** before full integration prevented potential training instability.

**Dead Ends:**
Assuming distillation and system optimizations are independent missed potential interactions between temperature scaling and mixed precision.

**Next Steps:**
Run full pipeline with all system optimizations plus Vanilla KD and record time-to-target for direct comparison.

---

Note: This summary was [written by us | helped by ChatGPT | verified by Claude]
