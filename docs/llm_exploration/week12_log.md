# Week 12 LLM Exploration Summary

## Session Focus
These conversations focused on **implementing Decoupled Knowledge Distillation (DKD)** and investigating whether separating target-class and non-target-class supervision could further accelerate Vision Transformer convergence on CIFAR-10. The discussions emphasized **loss decomposition, hyperparameter sensitivity, and optimization dynamics**.

---

## Conversation 1: DKD Loss Decomposition

**Prompt That Worked:**
"How does Decoupled Knowledge Distillation separate target-class and non-target-class supervision?"

**Key Insights:**
DKD splits the standard KL divergence into two components — TCKD focuses on the binary distribution between the target class and all other classes combined, while NCKD focuses exclusively on the distribution among non-target classes with the target masked out. This allows independent weighting of correctness versus dark knowledge signals.

**Techniques That Worked:**
Asking about **the mathematical decomposition of the KL divergence term** directly clarified how TCKD and NCKD relate to Vanilla KD.

**Dead Ends:**
Treating TCKD and NCKD as completely independent signals missed their interaction through the shared temperature parameter.

**Next Steps:**
Implement DKD with α and β as tunable hyperparameters and measure convergence improvements over Vanilla KD.

---

## Conversation 2: Hyperparameter Sensitivity in DKD

**Prompt That Worked:**
"How should we tune α, β, and temperature for DKD without running exhaustive grid searches?"

**Key Insights:**
Starting from reference values in the Megvii mdistiller implementation provides a strong baseline. β typically has more impact on convergence speed than α because non-target dark knowledge contributes more to early-stage representation learning. Temperature affects both TCKD and NCKD simultaneously.

**Techniques That Worked:**
Using **reference implementation hyperparameters as a starting point** before sweeping reduced tuning time significantly.

**Dead Ends:**
Treating all three hyperparameters as equally important initially led to an overly complex search space.

**Next Steps:**
Prioritize β tuning first, then temperature, then α for most efficient hyperparameter optimization.

---

## Conversation 3: Non-Target Dark Knowledge Contribution

**Prompt That Worked:**
"Why does non-target class supervision contribute more to convergence speed than target-class supervision?"

**Key Insights:**
The NCKD term provides rich relational structure among incorrect classes that helps the student build better decision boundaries early in training. The target class signal alone is relatively simple — the student already learns this from cross-entropy. The inter-class similarity structure is what accelerates representation learning.

**Techniques That Worked:**
Framing the question around **what information is unique to distillation versus what cross-entropy already provides** produced clearer insight.

**Dead Ends:**
Focusing on TCKD as the primary signal missed the stronger contribution of NCKD to early convergence.

**Next Steps:**
Run ablation comparing TCKD-only, NCKD-only, and full DKD to quantify individual contributions.

---

## Conversation 4: Feature Distillation as Next Direction

**Prompt That Worked:**
"Can matching intermediate representations provide stronger supervision than output logits alone?"

**Key Insights:**
Logit-based supervision provides only sparse 10-way class probability information per sample. Feature matching provides dense per-sample regression targets aligned with the teacher's internal representation space, which gives the student richer optimization signal at every layer.

**Techniques That Worked:**
Comparing **information density of logit supervision versus feature supervision** clarified why feature matching accelerates convergence beyond what DKD alone achieves.

**Dead Ends:**
Assuming logit supervision fully captures the teacher's knowledge missed the representation-level signal available through feature alignment.

**Next Steps:**
Implement feature distillation by aligning ResNet-56 penultimate features with ViT CLS token representations through a learnable linear projection.

---

## Conversation 5: DKD Implementation Validation

**Prompt That Worked:**
"How do we verify that the DKD implementation is numerically correct?"

**Key Insights:**
Key checks include confirming that TCKD and NCKD sum to approximately the same value as Vanilla KD under matched settings, verifying that the non-target mask correctly zeros out target class probability mass, and confirming loss values are finite and decreasing consistently.

**Techniques That Worked:**
Running **unit tests with synthetic logits** before full training validated correctness without requiring expensive GPU runs.

**Dead Ends:**
Relying only on training accuracy curves to validate correctness delayed detection of subtle implementation bugs.

**Next Steps:**
Add unit tests for loss functions to test suite and confirm numerical behavior under edge cases.

---

Note: This summary was [written by us | helped by ChatGPT | verified by Claude]
