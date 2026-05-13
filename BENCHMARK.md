# Fast ViT on CIFAR-10 — Benchmark Rules (v0.1)

This document defines the rules for the open Fast-ViT-on-CIFAR-10 benchmark and competition. See the project [`README.md`](README.md) for the reference implementation and the four reference runs.

The goal of the benchmark is simple:

> **Train a Vision Transformer to reach a target test accuracy on CIFAR-10 as fast as possible in wall-clock seconds.**

The benchmark explicitly invites — and has a dedicated track for — **distillation from a pretrained CNN teacher**, motivated by this project's finding that distillation substantially reduces time-to-target for small ViTs.

---

## 1. Tracks

A submission may enter one or both tracks. The leaderboard is ranked **independently per track**.

### Track A — No Distillation

Student-only training. **No external teacher model and no pretrained weights of any kind** may be used at any point in the timed run. ImageNet / CIFAR-100 / self-supervised checkpoints are all disallowed.

### Track B — Distillation

A frozen pretrained CNN teacher is allowed. **Exactly one** teacher checkpoint is permitted:

- `chenyaofo/pytorch-cifar-models / cifar10_resnet56`, loaded via `torch.hub`, at the commit SHA pinned in the benchmark repo.

No other pretrained weights may be used at any stage. The teacher may be frozen, partially frozen, or fine-tuned — but the fine-tune cost is included in `time_to_target` if it happens.

---

## 2. The fixed contract

The following are **non-negotiable** for a scored run.

### F1. Target metric

`time_to_target` = wall-clock seconds between the **first batch fed to the student's optimizer** and the **first validation step at which test accuracy ≥ TARGET_ACC**.

- **`TARGET_ACC = 85.0%`** in v0.1.
- Lower `time_to_target` is better.

### F2. Hardware

Final ranking is determined by the organizer re-running the submission on the **official SKU**:

- **1× NVIDIA H100 80GB SXM, single GPU**, on the cloud instance type published in the benchmark repo.
- Submitter-reported numbers are accepted for the public leaderboard but are **not** used for final standings.

### F3. Software base

The benchmark ships an **official Docker image** with pinned PyTorch / CUDA / cuDNN / Triton versions (`Dockerfile.benchmark` in the benchmark repo). Submissions:

- **must** build on the official base image,
- **may** install additional Python packages on top,
- **may not** modify the base image's CUDA, driver, kernel, or PyTorch versions.

### F4. Dataset

Standard CIFAR-10:

- 50,000 train / 10,000 test images, the official `torchvision` split.
- The test set **may not** be used during training for any purpose — including model selection, augmentation tuning, hyperparameter search, or early stopping.
- **No external labeled or unlabeled data** is allowed in either track.

### F5. Model size

Total trainable parameters in the **student at inference time** ≤ **5,000,000 (5M)**.

- Auxiliary modules used **only during training** (e.g., projection heads for feature distillation, EMA shadow copies) are excluded from this count.
- Such auxiliary modules must be **removed or disabled before the final test forward**; the test forward must run only the inference student.

### F6. Architectural constraint (student)

The student must be a Vision Transformer:

- Contains **≥ 4 transformer blocks**, each with self-attention + MLP + residual + LayerNorm.
- The classifier head is either a `[CLS]`-token readout or a global average pool over tokens.
- A **ConvStem before the transformer body is permitted**, provided it has **≤ 4 conv layers and ≤ 100k parameters**.
- Architectures whose forward pass is dominated by convolutional or recurrent operations (e.g., a pure CNN with a single attention block tacked on) are **not eligible**.

### F7. Evaluation protocol

- Validation runs on the **full 10,000-image** test set with the standard test normalization.
- Validation must occur **at least every 5 seconds of wall time** OR at the end of every epoch — whichever is more frequent.
- `time_to_target` is the **wall time at the first validation point where test accuracy ≥ TARGET_ACC**.

### F8. Seeds

Each submission must run with **3 distinct random seeds**.

- All 3 runs must reach `TARGET_ACC`; otherwise the submission is disqualified.
- The reported score for the leaderboard is the **median** of the three `time_to_target` values.
- Min and max are also published.

### F9. Wall-time cap

A single training run may not exceed **1 hour of wall time**. Runs that do not reach `TARGET_ACC` within the cap fail.

---

## 3. What is free

Within the fixed contract above, submissions may freely choose:

- **Loss function.** Cross-entropy, label smoothing, vanilla KD, DKD, feature distillation, attention transfer, relational KD, contrastive KD, any combination.
- **Optimizer and schedule.** AdamW, Muon, Lion, SGD-Nesterov, custom optimizers; any warmup + decay shape; gradient clipping; weight decay.
- **Augmentations.** RandAugment, TrivialAugment, AutoAugment, Mixup, CutMix, custom. Augmentations may be applied to the **training set only**.
- **Batch size, gradient accumulation, EMA, weight averaging, Polyak averaging.**
- **Precision and runtime.** fp32 / fp16 / bf16, `torch.compile`, custom CUDA / Triton kernels, FlashAttention, `cudnn.benchmark`, etc.
- **Architecture details within the ViT family.** Patch size, embed dim, depth, head count, MLP ratio, attention variant (e.g., SDPA, linear attention), normalization placement — subject to F5 and F6.
- **Whether to include a ConvStem** — subject to F6.

In **Track B**, additionally:

- **Distillation signal.** Logit-level, intermediate features, attention maps, self-supervised auxiliary objectives — any combination.
- **Teacher forward optimization.** Compile the teacher, run it in lower precision, cache its outputs across epochs — provided cached outputs are not used to leak the test set.

---

## 4. Time accounting

**Included in `time_to_target`:**

- Forward + backward + optimizer step of the student.
- Teacher forward (Track B), including any teacher fine-tuning step if used.
- Augmentation and data preprocessing on the training batch.
- Loss computation, including auxiliary heads (e.g., feature projection).
- Validation overhead at the official cadence (F7).
- **`torch.compile` first-call compilation time is included.** Paying the compile cost is part of training fast.

**Excluded from `time_to_target`:**

- CIFAR-10 download — must complete before the timer starts.
- Teacher checkpoint download (Track B) — must complete before the timer starts.
- Python process startup and CUDA initialization — timer starts at the **first optimizer step**.

---

## 5. Submission

A submission is a **public git repository** containing:

1. `Dockerfile` extending the official base image (or `requirements.txt` buildable on it).
2. `train.py` with the following CLI:

    ```bash
    python train.py --seed SEED \
                    --target_acc 85.0 \
                    --track {A,B} \
                    --data_dir DATA \
                    --out_dir OUT
    ```

3. `run.sh` — one command that builds the image and runs all 3 seeds, producing `OUT/run_seed{0,1,2}.json`.
4. `README.md` describing the method in 1–3 paragraphs.
5. `LICENSE` (a permissive open-source license is required to publish to the leaderboard).

Each `run_seedN.json` must follow the schema:

```json
{
  "track": "B",
  "seed": 0,
  "target_acc": 85.0,
  "time_to_target": 123.4,
  "best_acc": 81.9,
  "history": {
    "wall_time": [/* per-validation cumulative wall seconds */],
    "test_acc":  [/* per-validation test accuracy %         */]
  }
}
```

Length of `wall_time` and `test_acc` must match. The first entry at which `test_acc >= target_acc` must correspond to `time_to_target`.

---

## 6. Verification

When a submission is selected for the leaderboard:

1. The organizer clones the repo at the submitted commit SHA.
2. The organizer builds the image and runs `run.sh` on the official SKU.
3. The organizer reloads the saved student checkpoint and **re-evaluates** test accuracy. The submission is accepted only if the recomputed accuracy at the claimed `time_to_target` step is within **±0.2%** of the JSON's reported value.
4. The test tensor's hash is asserted to be the standard CIFAR-10 test split before any timed run.
5. Each timed run is monitored for total wall time; any run exceeding 1 hour is auto-failed.

Submissions that fail any verification step are removed from the leaderboard and the authors are contacted before publication.

---

## 7. Leaderboard

For each track:

| Rank | Method | Median `time_to_target` (s) | Min / Max | Best test acc (%) | Repo | Image SHA |
|------|--------|-----------------------------|-----------|--------------------|------|-----------|

Entries are sorted by median `time_to_target` ascending. Ties broken by min `time_to_target`.

Each entry links to the public repo, the pinned commit SHA, and the exact Docker image SHA used by the organizer's verifying run.

---

## 8. Open design questions (subject to community input before v1.0)

These items are deliberately left open for v0.1 and will be revisited:

- Whether to raise `TARGET_ACC` to 90% or 94% in a higher-difficulty track.
- Whether Track B should allow **CIFAR-100** or **ImageNet** pretrained teachers, possibly as a separate track.
- Whether to add a Track C with an explicit **GPU-hours / dollar-cost** cap as an additional dimension.
- Whether to allow **multiple teachers** in Track B.

Proposals for revisions are welcome via issues / PRs in the benchmark repo.

---

## 9. License and provenance

This rules document is released under the same license as the rest of the project. The reference implementation provided in `notebooks/no_distillation_baseline.ipynb` (Track A reference) and `notebooks/feature_distillation.ipynb` (Track B reference) serves as the **canonical starting point** for participants; both produce JSON logs that already conform to the schema in §5.
