import { ModelInfo, PredictionResult } from "@/types/prediction";

const RAW_API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api";
const CLEAN_URL = RAW_API_URL.replace(/\/+$/, "");
const API_BASE_URL = CLEAN_URL.endsWith("/api") ? CLEAN_URL : `${CLEAN_URL}/api`;

export async function fetchHealth(): Promise<{ status: string; checkpoints_available?: boolean }> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      signal: AbortSignal.timeout ? AbortSignal.timeout(2000) : undefined,
    });
    if (!res.ok) {
      throw new Error(`Health check returned ${res.status}`);
    }
    return await res.json();
  } catch (error) {
    console.warn("Backend health check failed:", error);
    return { status: "offline", checkpoints_available: false };
  }
}

export async function fetchModels(): Promise<ModelInfo[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/models`, {
      signal: AbortSignal.timeout ? AbortSignal.timeout(3000) : undefined,
    });
    if (!res.ok) {
      throw new Error(`Failed to fetch models: ${res.statusText}`);
    }
    return await res.json();
  } catch (error) {
    console.warn("Backend not available, using fallback model list", error);
    return [];
  }
}

export async function runPrediction(
  imageFile: File | Blob,
  modelId: string,
  explain: boolean = true
): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", imageFile, "leaf_image.jpg");
  formData.append("model_id", modelId);
  formData.append("explain", String(explain));

  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || "Prediction request failed.");
  }

  return await res.json();
}
