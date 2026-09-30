export interface SampleLeaf {
  id: string;
  name: string;
  category: "Early Blight" | "Late Blight" | "Healthy";
  description: string;
  path: string;
  pathogen: string;
}

export const SAMPLE_LEAVES: SampleLeaf[] = [
  {
    id: "sample-early-blight",
    name: "Early Blight Leaf Sample",
    category: "Early Blight",
    description: "Characteristic concentric target-board ring lesions caused by Alternaria solani fungus.",
    path: "/samples/sample_early_blight.jpg",
    pathogen: "Alternaria solani",
  },
  {
    id: "sample-late-blight",
    name: "Late Blight Leaf Sample",
    category: "Late Blight",
    description: "Water-soaked irregular necrotic brown lesions caused by Phytophthora infestans oomycete.",
    path: "/samples/sample_late_blight.jpg",
    pathogen: "Phytophthora infestans",
  },
  {
    id: "sample-healthy",
    name: "Healthy Potato Foliage",
    category: "Healthy",
    description: "Asymptomatic potato leaf with uniform chlorophyll pigmentation and intact leaf veins.",
    path: "/samples/sample_healthy.jpg",
    pathogen: "None (Healthy)",
  },
];
