# Project Documentation

This directory contains research and technical documentation for the **Potato Disease Research Demo**.

## Contents
- **Research Methodology:** Details on the dataset splits (PlantVillage controlled environment vs. PLDD-UP natural field environment), cross-dataset evaluation protocol, and generalization gap metrics.
- **Model Architectures:**
  1. **EfficientNet-B0**: Baseline CNN utilizing mobile inverted bottleneck convolutions (MBConv) and compound scaling.
  2. **DeiT-Tiny**: Compact vision transformer architecture utilizing distillation tokens and self-attention.
  3. **CNN–ViT Concat Fusion**: Direct concatenation of global average pooled CNN feature maps and ViT class tokens.
  4. **Attention-Guided CNN–ViT Fusion**: Proposed mechanism using cross-attention or attention-weighted gating between local CNN texture representations and global ViT contextual representations.
- **Explainability Pipelines:** Grad-CAM for CNN activation localization and attention weight extraction for transformer/fusion layers.
