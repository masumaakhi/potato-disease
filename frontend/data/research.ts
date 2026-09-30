export const RESEARCH_METADATA = {
  title: "An Attention-Guided Lightweight CNN–ViT Fusion Network for Potato Leaf Disease Classification Across Controlled and Natural Environments",
  shortTitle: "Attention-Guided Lightweight CNN–ViT Fusion",
  classes: [
    { name: "Potato Early Blight", pathogen: "Alternaria solani", severity: "High" },
    { name: "Potato Late Blight", pathogen: "Phytophthora infestans", severity: "Critical" },
    { name: "Healthy Potato Leaf", pathogen: "None", severity: "None" },
  ],
  datasets: [
    {
      name: "PlantVillage",
      environment: "Controlled Laboratory",
      description: "Standardized background, uniform illumination, isolated single-leaf imagery.",
    },
    {
      name: "PLDD-UP",
      environment: "Natural Field Environment",
      description: "Uncontrolled field conditions, complex background noise, variable lighting, multi-leaf clusters.",
    },
  ],
  architectures: [
    {
      name: "EfficientNet-B0",
      type: "CNN Baseline",
      focus: "Local fine-grained spatial lesion patterns",
    },
    {
      name: "DeiT-Tiny",
      type: "ViT Baseline",
      focus: "Global contextual semantic relationships",
    },
    {
      name: "CNN-ViT Concat",
      type: "Fusion Baseline",
      focus: "Unweighted feature concatenation",
    },
    {
      name: "Attention-Guided Fusion",
      type: "Proposed Model",
      focus: "Cross-attention guided local-global synergy",
    },
  ],
};
