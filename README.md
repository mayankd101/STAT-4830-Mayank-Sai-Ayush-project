# STAT 4830: ViT Training Benchmark — Distillation from a CNN Teacher

A reproducible CIFAR-10 Vision Transformer training harness whose **primary metric is wall-clock time to a target test accuracy**, not just final accuracy. The headline experiment compares a small ViT trained from scratch against the same ViT distilled from a frozen pretrained ResNet-56, with three distillation variants of increasing strength.

> **Open benchmark.** This repository is also the reference implementation for an open *Fast ViT on CIFAR-10* benchmark with two tracks (no-distillation and distillation). The full ruleset — fixed hardware, dataset, eval protocol, time accounting, submission schema, and verification — lives in [`BENCHMARK.md`](BENCHMARK.md). The four runs in the "Reported benchmark" section below are the canonical reference numbers for both tracks.

## Reported benchmark

All four runs train the **same ViT student** (patch 4, embed 192, depth 6, heads 6) on CIFAR-10 with the **same** data loader, augmentations, batch size, LR schedule, validation cadence, and **target = 85% test accuracy**. The only thing that changes is the **loss**.

| Method | Notebook | Loss | Saved JSON tag |
|---|---|---|---|
| No distillation (baseline) | `notebooks/no_distillation_baseline.ipynb` | CE only (label smoothing 0.1) | `baseline` |
| Vanilla KD | `notebooks/kd_teacher_comparison.ipynb` | CE + KL on full teacher softmax | `vanilla_kd` |
| Decoupled KD (DKD) | `notebooks/dkd_distillation.ipynb` | CE + α·TCKD + β·NCKD | `dkd` |
| DKD + Feature distillation | `notebooks/feature_distillation.ipynb` | CE + DKD + MSE(proj(student CLS), teacher penult.) | `feat_dkd` |

Teacher = `chenyaofo/pytorch-cifar-models/cifar10_resnet56`, loaded via `torch.hub` and **frozen**.

---

## 1. Setup

### Hardware

- **Recommended:** a single CUDA GPU (the reported runs used an NVIDIA **GH200 96GB** on Lambda Cloud / Prime Intellect). Any modern Hopper or Ampere GPU should work.
- **Other:** Apple Silicon (`mps`) and CPU are auto-detected by `src.utils.get_device`, but per-epoch times are not representative.

### Environment

From the project root on the machine where you will train:

```bash
chmod +x setup_env.sh
./setup_env.sh
```

This creates `.venv`, installs `requirements.txt` (PyTorch, torchvision, timm, numpy, matplotlib, pytest), and registers a Jupyter kernel named **STAT4830 ViT (Python 3)**.

Activate the venv whenever you open a fresh shell:

```bash
source .venv/bin/activate
```

Sanity check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Notes on `torch.compile`

Some notebooks call `torch.compile`. It is gated by an import check — if **Triton is not installed or is incompatible with your PyTorch version** (common on aarch64 GH200 images), compile is **skipped automatically** with a printed message, and training continues in eager mode. No action needed.

If you want compile to be active and `pip install triton` fails on your arch, simply leave the guard in place; eager mode is what was used for the reported numbers anyway.

---

## 2. Run Jupyter

From the project root:

```bash
source .venv/bin/activate
jupyter lab --no-browser --ip=0.0.0.0 --port 8888
```

If you are SSH-ing into a remote GPU box, forward the port from your laptop in a second shell:

```bash
ssh -L 8888:127.0.0.1:8888 <user>@<host>
```

Then open `http://127.0.0.1:8888/lab` and paste the **token** printed by the Jupyter server. In the Lab UI, navigate to `notebooks/` and select kernel **STAT4830 ViT (Python 3)** (or the `.venv` kernel).

---

## 3. Reproducing the four benchmark runs

For each method below: open the notebook, **Run All**, wait, then **run the save cell at the bottom** to write a JSON the combined plot reads. CIFAR-10 is downloaded automatically on first run into `./data/`.

### a. Baseline (no distillation)

```text
notebooks/no_distillation_baseline.ipynb
```

Trains the ViT student with cross-entropy only (label smoothing 0.1). Writes `notebooks/run_baseline.json`.

### b. Vanilla KD

```text
notebooks/kd_teacher_comparison.ipynb
```

Trains the same student with CE + KL on the full teacher softmax. Writes `notebooks/run_vanilla_kd.json`.

### c. DKD

```text
notebooks/dkd_distillation.ipynb
```

DKD loss (Megvii reference): CE + α·TCKD + β·NCKD with a warm-up ramp. Writes `notebooks/run_dkd.json`.

### d. DKD + Feature distillation

```text
notebooks/feature_distillation.ipynb
```

Layered on top of DKD: a small `Linear(192 → 64)` projection aligns the student's `[CLS]` token (post-`LayerNorm`, pre-`head`) with the teacher's penultimate features (output of ResNet-56's `avgpool`); loss = `MSE(proj(student), teacher.detach())`. Writes `notebooks/run_feat_dkd.json`.

### Save-cell template

If you need to (re)create the save cell at the bottom of any of the four notebooks, change `TAG` and run:

```python
import json

TAG = "baseline"   # → "vanilla_kd", "dkd", "feat_dkd"
out = {
    "tag": TAG,
    "target_acc": TARGET_ACC,
    "time_to_target": results["time_to_target"],
    "best_acc": results["best_acc"],
    "history": {
        "test_acc": list(results["history"]["test_acc"]),
        "wall_time": list(results["history"]["wall_time"]),
    },
}
with open(PROJECT_ROOT / "notebooks" / f"run_{TAG}.json", "w") as f:
    json.dump(out, f)
print("saved", TAG)
```

---

## 4. Combined comparison plot

After running any subset of the four notebooks, run this cell from any notebook to produce the four-curve **wall-clock vs test accuracy** figure used in the report:

```python
import json
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
NB_DIR = PROJECT_ROOT / "notebooks"

RUNS = [
    ("baseline",   "No distillation (CE)", "#808080"),
    ("vanilla_kd", "Vanilla KD",           "#0077b6"),
    ("dkd",        "DKD",                  "#e07a00"),
    ("feat_dkd",   "DKD + Feature",        "#7b3ff2"),
]

fig, ax = plt.subplots(figsize=(7, 4.5))
target = None
for tag, label, color in RUNS:
    p = NB_DIR / f"run_{tag}.json"
    if not p.exists():
        print("skip", p.name)
        continue
    d = json.load(open(p))
    target = d["target_acc"]
    wt, acc = d["history"]["wall_time"], d["history"]["test_acc"]
    cum = [sum(wt[: i + 1]) for i in range(len(wt))]
    ax.plot(cum, acc, lw=2, color=color, label=label)
    if d["time_to_target"]:
        ax.axvline(d["time_to_target"], color=color, ls=":", alpha=0.6)
if target is not None:
    ax.axhline(target, color="gray", ls="--", label=f"target {target:g}%")
ax.set_xlabel("wall time (s)")
ax.set_ylabel("test %")
ax.set_title("CIFAR-10 ViT: time-to-target across methods")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(NB_DIR / "all_methods_wallclock.png", dpi=150, bbox_inches="tight")
plt.show()
```

The cell silently skips any method whose JSON is not present, so it is safe to run incrementally as you finish each method.

---

## 5. Project layout

```text
src/
  model.py            ViT (patch embed, MHA, MLP, transformer blocks, CLS token classifier).
  utils.py            CIFAR-10 loaders, device selection, seeding, train/validate,
                      AMP helpers, GPU memory metrics, cosine-with-warmup schedule,
                      torch.compile support.
  losses.py           Loss functions: CE, Vanilla KD, DKD, DKD + feature distillation.

notebooks/
  no_distillation_baseline.ipynb     Baseline: CE-only ViT (target 85%).
  kd_teacher_comparison.ipynb        Vanilla KD across teacher architectures (target 85%).
  dkd_distillation.ipynb             DKD (target 85%).
  feature_distillation.ipynb         DKD + feature matching (target 85%).
  vit_baseline_implementation.ipynb  Week 4: initial ViT pipeline.
  vit_pipeline_optimization.ipynb    Week 6: wall-clock pipeline tuning.
  vit_advanced_optimizations.ipynb   Week 8: bf16 / compile / Muon / mixup ablations.
  cnn_teacher_distillation.ipynb     Earlier CNN-teacher distillation experiments.

docs/
  development_log.md                 Self-critiques and weekly notes (Weeks 4–12).
  llm_exploration/                   LLM-assisted exploration notes.

self_critiques/                      Self-critiques are also here individually
old_reports/                         Old report drafts from weeks 4-12
old_slides_drafts/                   Old slide drafts from weeks 4-12

demo.ipynb            Executable demo

final_report.md       Final report file
final_slides.pdf      Final presentation slides file

tests/test_basic.py   Smoke tests for model and utilities.
requirements.txt      Top-level pinned dependencies.
setup_env.sh          One-shot venv + dependency + kernel installer.
```

---

## 6. Reproducibility notes

- **Seed.** All four headline notebooks call `set_seed(42, deterministic=False)`. We use `cudnn.benchmark = True` for speed; reported numbers therefore have **non-zero run-to-run variance**, especially the precise `time_to_target`. For a tighter estimate, run each method with **≥ 2 seeds** and report the median.
- **Hardware variance.** `time_to_target` depends on GPU, host CPU, NVMe vs networked storage, and PyTorch / CUDA / Triton versions. Absolute wall-clock numbers should only be compared **within the same machine and software stack**.
- **Eval cadence.** Validation runs every 2 epochs until test accuracy crosses `VALIDATE_DENSE_AFTER = 74.0%`, then every epoch. This is identical across all four notebooks so threshold-crossing timing is comparable.
- **What we do not vary across the four runs.** Student architecture, optimizer (AdamW, lr `1e-3` linear-scaled from `1e-3 @ batch 512`, weight decay `0.01`), cosine-with-warmup schedule (5 warm-up epochs), batch size, RandAugment configuration, bf16 autocast on CUDA, and gradient clip 1.0.
- **What we do vary (the experiment).** Only the **loss**: CE → CE+KL → CE+DKD → CE+DKD+feature.

---

## 7. Project overview

This project builds a modded-nanoGPT–style training harness for Vision Transformers. Instead of optimizing only for final accuracy, we optimize for **time-to-target accuracy** — how fast (in wall-clock seconds) a model can reach a specified performance threshold on CIFAR-10.

The headline finding is that **distillation from a pretrained CNN teacher reduces this time** for a small ViT, with **DKD + feature matching** producing the largest improvement over the no-distillation baseline. Earlier notebooks document the systems-side path that led to this conclusion: data pipeline tuning, AMP / bf16, schedule design, and earlier distillation experiments.
