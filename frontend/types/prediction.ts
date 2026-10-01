export interface DiseaseClass {
  id: number;
  name: string;
  code: string;
  pathogen?: string;
  severity?: "Low" | "Moderate" | "High" | "Critical" | "None";
  description?: string;
}

export interface ClassProbability {
  class_name: string;
  probability: number;
  percentage: string;
  class_id?: number;
}

export interface AttentionWeightsData {
  cnn_branch_weight: number;
  vit_branch_weight: number;
  dominant_branch: "CNN" | "ViT" | string;
}

export interface InferenceInfoData {
  inference_time_ms: number;
  device: string;
  input_resolution: string;
  timestamp: string;
}

export interface GradCAMViewData {
  diagnostic_label: string;
  disclaimer: string;
  model_id: string;
  model_name: string;
  target_layer: string;
  predicted_class: string;
  predicted_class_id: number;
  confidence: number;
  target_class: string;
  target_class_id: number;
  target_class_probability: number;
  gradcam_image_base64: string;
  heatmap_image_base64?: string | null;
  heatmap_grid_size?: number[];
  resolution?: string;
  alpha_blend?: number;
}

export interface PredictionResult {
  model_name: string;
  model_id: string;
  predicted_class: string;
  predicted_class_id?: number;
  confidence: number;
  probabilities: ClassProbability[];
  inference_time_ms: number;
  inference_info?: InferenceInfoData | null;
  attention_weights?: AttentionWeightsData | null;
  attention_data?: AttentionWeightsData | Record<string, unknown> | null;
  gradcam_available: boolean;
  attention_weights_available: boolean;
  gradcam_image_base64?: string | null;
  gradcam_view?: GradCAMViewData | null;
  specimen_warning?: string | null;
  is_likely_potato_leaf?: boolean;
  specimen_status?: "optimal" | "synthetic_advisory" | "low_confidence_advisory" | string;
}

export interface ModelInfo {
  id: string;
  name: string;
  architecture: string;
  description: string;
  parameters?: string;
  checkpoint_available: boolean;
}

export interface MetricEntry {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  accuracy_std?: number;
  f1_score_std?: number;
  params_m?: number;
  flops_g?: number;
  latency_ms?: number;
  throughput_fps?: number;
  weights_mb?: number;
}
