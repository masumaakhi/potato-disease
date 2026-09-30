import { ModelInfo } from "@/types/prediction";

export const RESEARCH_MODELS: ModelInfo[] = [
  {
    id: "attention_fusion",
    name: "Attention-Guided CNN–ViT (Proposed)",
    architecture: "Lightweight Cross-Attention Network",
    description: "Proposed adaptive model learning dynamic softmax-normalized attention weights [α_cnn, α_vit] to synergize local convolutional lesion patterns with global ViT context.",
    parameters: "10.11M",
    checkpoint_available: true,
  },
  {
    id: "efficientnet_b0",
    name: "EfficientNet-B0 (CNN Baseline)",
    architecture: "Lightweight Inverted Bottleneck CNN",
    description: "Convolutional baseline utilizing depthwise separable convolutions and squeeze-and-excitation blocks for fine-grained local spatial texture extraction.",
    parameters: "4.01M",
    checkpoint_available: true,
  },
  {
    id: "deit_tiny",
    name: "DeiT-Tiny (ViT Baseline)",
    architecture: "Compact Vision Transformer",
    description: "Transformer baseline utilizing patch projection (16x16) and multi-head self-attention to model long-range context without convolutional inductive biases.",
    parameters: "5.52M",
    checkpoint_available: true,
  },
  {
    id: "cnn_vit_concat",
    name: "CNN–ViT Concatenation (Baseline)",
    architecture: "Direct Feature Concatenation",
    description: "Standard fusion baseline concatenating pooled EfficientNet-B0 spatial features (1280d) with DeiT-Tiny class tokens (192d) into a shared MLP classifier.",
    parameters: "10.29M",
    checkpoint_available: true,
  },
];
