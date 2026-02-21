# STAT 4830: Modded ViT Training Pipeline

Vision Transformer training harness for CIFAR-10, focused on wall-clock time to target accuracy.

## Quick setup (run once)

From the project root:

```bash
chmod +x setup_env.sh
./setup_env.sh
```

This creates a `.venv`, installs PyTorch and dependencies, and registers a Jupyter kernel.

## Running the notebook

1. Open `notebooks/week6_optimization.ipynb`
2. Click the kernel selector (top-right)
3. Choose **"STAT4830 ViT (Python 3)"** (after running setup)  
   Or choose **".venv (Python)"** / the interpreter at `STAT4830/.venv/bin/python`

## Kernel = which Python runs your code

The kernel is the Python environment that executes notebook cells. Pick one that has `torch` and `torchvision` installed (the `.venv` after setup).

## Project Overview

This project builds a modded-nanoGPT–style training harness for Vision Transformers (ViTs).
Instead of optimizing only for final accuracy, we optimize for time-to-target accuracy — i.e., how fast (in wall-clock seconds) a model can reach a specified performance threshold on a standard vision benchmark such as CIFAR-10.

The core idea is to treat the entire training pipeline — architecture, optimizer, batch size, precision, augmentations, and data loading — as a “recipe” that can be systematically modified and benchmarked. Inspired by modded-nanogpt and cifar10-airbench, this repository provides a clean, reproducible baseline implementation and a measurement framework to evaluate which changes actually make training faster.
