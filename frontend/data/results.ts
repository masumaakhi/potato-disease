import { MetricEntry } from "@/types/prediction";

/**
 * Authentic empirical benchmark results from the research paper:
 * "An Attention-Guided Lightweight CNN–ViT Fusion Network for Potato Leaf
 * Disease Classification Across Controlled and Natural Environments"
 * Evaluated across 3 random seeds (seed 42, 123, 456).
 */

export const INTERNAL_TEST_RESULTS: MetricEntry[] = [
  {
    model: "CNN Baseline (EfficientNet-B0)",
    accuracy: 1.0000,
    accuracy_std: 0.0000,
    precision: 1.0000,
    recall: 1.0000,
    f1_score: 1.0000,
    f1_score_std: 0.0000,
    params_m: 4.01,
    flops_g: 0.40,
    weights_mb: 15.46,
    latency_ms: 19.40,
    throughput_fps: 51.55,
  },
  {
    model: "ViT Baseline (DeiT-Tiny)",
    accuracy: 1.0000,
    accuracy_std: 0.0000,
    precision: 1.0000,
    recall: 1.0000,
    f1_score: 1.0000,
    f1_score_std: 0.0000,
    params_m: 5.52,
    flops_g: 1.08,
    weights_mb: 21.08,
    latency_ms: 5.51,
    throughput_fps: 181.40,
  },
  {
    model: "CNN–ViT Concatenation",
    accuracy: 1.0000,
    accuracy_std: 0.0000,
    precision: 1.0000,
    recall: 1.0000,
    f1_score: 1.0000,
    f1_score_std: 0.0000,
    params_m: 10.29,
    flops_g: 1.48,
    weights_mb: 39.42,
    latency_ms: 13.30,
    throughput_fps: 75.18,
  },
  {
    model: "Attention-Guided CNN–ViT (Proposed)",
    accuracy: 0.9990,
    accuracy_std: 0.0018,
    precision: 0.9993,
    recall: 0.9993,
    f1_score: 0.9993,
    f1_score_std: 0.0013,
    params_m: 10.11,
    flops_g: 1.48,
    weights_mb: 38.72,
    latency_ms: 13.59,
    throughput_fps: 73.59,
  },
];

export const EXTERNAL_TEST_RESULTS: MetricEntry[] = [
  {
    model: "CNN Baseline (EfficientNet-B0)",
    accuracy: 0.4999,
    accuracy_std: 0.0378,
    precision: 0.5626,
    recall: 0.5275,
    f1_score: 0.4866,
    f1_score_std: 0.0506,
    params_m: 4.01,
    flops_g: 0.40,
    weights_mb: 15.46,
    latency_ms: 19.40,
    throughput_fps: 51.55,
  },
  {
    model: "ViT Baseline (DeiT-Tiny)",
    accuracy: 0.6352,
    accuracy_std: 0.0215,
    precision: 0.7006,
    recall: 0.6155,
    f1_score: 0.6238,
    f1_score_std: 0.0311,
    params_m: 5.52,
    flops_g: 1.08,
    weights_mb: 21.08,
    latency_ms: 5.51,
    throughput_fps: 181.40,
  },
  {
    model: "CNN–ViT Concatenation",
    accuracy: 0.6157,
    accuracy_std: 0.0799,
    precision: 0.6884,
    recall: 0.5996,
    f1_score: 0.5903,
    f1_score_std: 0.1264,
    params_m: 10.29,
    flops_g: 1.48,
    weights_mb: 39.42,
    latency_ms: 13.30,
    throughput_fps: 75.18,
  },
  {
    model: "Attention-Guided CNN–ViT (Proposed)",
    accuracy: 0.5685,
    accuracy_std: 0.0578,
    precision: 0.6677,
    recall: 0.5430,
    f1_score: 0.5374,
    f1_score_std: 0.0838,
    params_m: 10.11,
    flops_g: 1.48,
    weights_mb: 38.72,
    latency_ms: 13.59,
    throughput_fps: 73.59,
  },
];

export interface GeneralizationGapEntry {
  model: string;
  internal_accuracy: number;
  external_accuracy: number;
  accuracy_drop_pct: number;
  internal_macro_f1: number;
  external_macro_f1: number;
  macro_f1_drop_pct: number;
}

export const GENERALIZATION_GAP_DATA: GeneralizationGapEntry[] = [
  {
    model: "EfficientNet-B0",
    internal_accuracy: 100.0,
    external_accuracy: 49.99,
    accuracy_drop_pct: 50.01,
    internal_macro_f1: 100.0,
    external_macro_f1: 48.66,
    macro_f1_drop_pct: 51.34,
  },
  {
    model: "DeiT-Tiny",
    internal_accuracy: 100.0,
    external_accuracy: 63.52,
    accuracy_drop_pct: 36.48,
    internal_macro_f1: 100.0,
    external_macro_f1: 62.38,
    macro_f1_drop_pct: 37.62,
  },
  {
    model: "CNN–ViT Concatenation",
    internal_accuracy: 100.0,
    external_accuracy: 61.57,
    accuracy_drop_pct: 38.43,
    internal_macro_f1: 100.0,
    external_macro_f1: 59.03,
    macro_f1_drop_pct: 40.97,
  },
  {
    model: "Attention-Guided CNN–ViT",
    internal_accuracy: 99.90,
    external_accuracy: 56.85,
    accuracy_drop_pct: 43.05,
    internal_macro_f1: 99.93,
    external_macro_f1: 53.74,
    macro_f1_drop_pct: 46.19,
  },
];

export interface EfficiencyEntry {
  model: string;
  parameters: number;
  params_formatted: string;
  weights_mb: number;
  gflops: number;
  latency_ms: number;
  throughput_fps: number;
  gpu_memory_mb: number;
}

export const EFFICIENCY_COMPARISON_DATA: EfficiencyEntry[] = [
  {
    model: "EfficientNet-B0",
    parameters: 4011391,
    params_formatted: "4.01M",
    weights_mb: 15.46,
    gflops: 0.40,
    latency_ms: 19.40,
    throughput_fps: 51.55,
    gpu_memory_mb: 35.19,
  },
  {
    model: "DeiT-Tiny",
    parameters: 5524995,
    params_formatted: "5.52M",
    weights_mb: 21.08,
    gflops: 1.08,
    latency_ms: 5.51,
    throughput_fps: 181.40,
    gpu_memory_mb: 32.53,
  },
  {
    model: "CNN–ViT Concatenation",
    parameters: 10290623,
    params_formatted: "10.29M",
    weights_mb: 39.42,
    gflops: 1.48,
    latency_ms: 13.30,
    throughput_fps: 75.18,
    gpu_memory_mb: 59.17,
  },
  {
    model: "Attention-Guided CNN–ViT (Proposed)",
    parameters: 10109249,
    params_formatted: "10.11M",
    weights_mb: 38.72,
    gflops: 1.48,
    latency_ms: 13.59,
    throughput_fps: 73.59,
    gpu_memory_mb: 58.48,
  },
];
