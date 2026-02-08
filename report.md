# Week 4: Vision Transformer (ViT) Training Pipeline for CIFAR-10

## Problem Statement

**Goal:** Build a reproducible Vision Transformer (ViT) training pipeline for CIFAR-10, optimizing **wall-clock time to reach 94% test accuracy**.  

**Optimization Target:** Minimize **time-to-target accuracy** while monitoring throughput (images/sec) and GPU memory usage.  

**Motivation:** Efficient training pipelines allow us to explore model modifications quickly, enabling faster iteration and benchmarking for future research. CIFAR-10 is a standard benchmark dataset that provides a controlled environment for testing both pre-trained and scratch ViT models.  

**Success Metrics:**  
- **Primary:** Time (seconds) to reach 94% test accuracy  
- **Secondary:** Training/test accuracy curves, throughput (images/sec), peak GPU memory  

**Constraints:**  
- Runs must be reproducible (fixed random seed)  
- Resource-limited (GPU memory, batch size)  
- Pre-trained ViT requires image resizing and normalization  

**Data Requirements:** CIFAR-10 dataset (50k train / 10k test, 32×32 RGB, 10 classes). For pre-trained ViT, images are resized to 224×224 and normalized using ImageNet statistics.  

**Potential Risks:**  
- Target accuracy may not be reached due to limited epochs or suboptimal hyperparameters  
- Hardware constraints (GPU memory) could limit batch size  
- Deterministic settings may slow training or affect throughput  

---

## Technical Approach

**Mathematical Formulation:**  

\[
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log \hat{p}_{i,c}
\]  

- \(y_{i,c}\): one-hot labels  
- \(\hat{p}_{i,c}\): predicted softmax probability for class \(c\)  
- Objective: minimize cross-entropy loss while achieving target test accuracy  

**Algorithm Choice:**  
- Vision Transformer (ViT), either pre-trained on ImageNet or trained from scratch  
- Pre-trained model provides faster convergence; scratch model allows ablation studies  
- Transformer architecture chosen for ability to process image patches and model global relationships  

**PyTorch Implementation Strategy:**  
- Use modular code in `src/` for model and utilities  
- Notebook `notebooks/week4_implementation.ipynb` handles data loading, training, validation, and plotting  
- Data loaders implement augmentation and resizing  
- Early stopping triggered upon reaching 94% accuracy  
- Training loop logs throughput, peak GPU memory, and epoch accuracy  

**Validation Methods:**  
- Test output shapes and loss finiteness  
- Reproducibility verified via fixed seed  
- Accuracy curves plotted to monitor performance  
- Peak GPU memory recorded per epoch  

**Resource Requirements:**  
- GPU recommended (MPS or CUDA supported)  
- Moderate memory footprint (batch sizes 64–128 depending on model)  
- Wall-clock time monitored for optimization  

---

## Initial Results

**Evidence of Implementation Success:**  
- Notebook successfully runs a forward pass on a sample batch  
- Model outputs match expected shapes `(B, 10)`  

**Performance Metrics:**  
- Pre-trained ViT: ~5.5M parameters  
- Batch size: 64  
- Time-to-target: recorded in training logs  
- Throughput: tracked per epoch  
- Peak GPU memory usage: logged  

**Test Case Results:**  
- Output shape: `(batch_size, 10)`  
- Loss: finite and stable  
- Training reproducible across runs  

**Current Limitations:**  
- 10–20 epochs may not reach 94% accuracy  
- Deterministic mode slows training  
- Augmentations are basic (RandomCrop, HorizontalFlip); more advanced methods could improve performance  

**Resource Usage:**  
- Moderate GPU memory  
- Wall-clock training time depends on pre-trained vs scratch  

**Unexpected Challenges:**  
- Early stopping integration and metric logging  
- Pre-trained model resizing and normalization  

---

## Next Steps

**Immediate Improvements:**  
- Implement full early stopping to capture exact time-to-target  
- Tune hyperparameters to reach 94% accuracy consistently  
- Add more advanced augmentations (Mixup, RandAugment)  

**Technical Challenges to Address:**  
- Optimizing memory and throughput for larger batch sizes  
- Integrating configuration files (YAML/JSON) for mod experiments  
- Mixed precision training (AMP) to reduce wall-clock time  

**Questions:**  
- Optimal patch size and depth for CIFAR-10  
- Best augmentation strategy for faster convergence  

**Alternative Approaches:**  
- Experiment with smaller ViT variants for speed  
- Compare scratch vs pre-trained performance  
- Use learning rate schedulers or adaptive optimizers  

**Key Learnings:**  
- Modular repo structure simplifies experimentation  
- Pre-trained models drastically reduce time-to-target  
- Accurate resource logging is crucial for reproducible benchmarking  

