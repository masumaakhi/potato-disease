"use client";

import React, { useState, useEffect } from "react";
import { RESEARCH_MODELS } from "@/data/models";
import { ModelSelector } from "@/components/prediction/model-selector";
import { ImageUploader } from "@/components/prediction/image-uploader";
import { PredictionCard } from "@/components/prediction/prediction-card";
import { ProbabilityChart } from "@/components/prediction/probability-chart";
import { AttentionWeights } from "@/components/prediction/attention-weights";
import { GradCAMViewer } from "@/components/prediction/gradcam-viewer";
import { PredictionResult } from "@/types/prediction";
import { runPrediction, fetchHealth } from "@/lib/api";
import { Play, AlertCircle, Wifi, WifiOff, Loader2 } from "lucide-react";

export default function PredictPage() {
  const [selectedModelId, setSelectedModelId] = useState<string>(RESEARCH_MODELS[0].id);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [explainEnabled, setExplainEnabled] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");

  useEffect(() => {
    async function checkStatus() {
      const health = await fetchHealth();
      setBackendStatus(health.status === "healthy" ? "online" : "offline");
    }
    checkStatus();
  }, []);

  const handleImageSelected = (file: File) => {
    setSelectedImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleClearImage = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (!selectedImage) {
      setError("Please select or upload a potato leaf image first.");
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const pred = await runPrediction(selectedImage, selectedModelId, explainEnabled);
      setResult(pred);
    } catch (err: unknown) {
      const msg =
        err instanceof Error
          ? err.message
          : "Prediction failed. Ensure the FastAPI backend server is running on http://127.0.0.1:8000.";
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 transition-colors">
      {/* Page Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-zinc-200 dark:border-zinc-800 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100 sm:text-3xl">
            Model Prediction & Diagnostic Workbench
          </h1>
          <p className="mt-1.5 text-xs sm:text-sm text-zinc-600 dark:text-zinc-400">
            Real-time inference using trained checkpoints with localized CNN Grad-CAM saliency and cross-attention weight gating.
          </p>
        </div>

        {/* Backend Connectivity Status Pill */}
        <div className="flex items-center gap-2 rounded-full border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/80 px-3 py-1.5 text-xs shadow-sm">
          {backendStatus === "online" ? (
            <>
              <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
              <Wifi className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
              <span className="text-zinc-700 dark:text-zinc-300 font-mono text-[11px] font-medium">Backend API Online</span>
            </>
          ) : backendStatus === "offline" ? (
            <>
              <span className="h-2 w-2 rounded-full bg-red-500" />
              <WifiOff className="h-3.5 w-3.5 text-red-500" />
              <span className="text-red-600 dark:text-red-300 font-mono text-[11px] font-medium">Backend Offline (Port 8000)</span>
            </>
          ) : (
            <span className="text-zinc-500 font-mono text-[11px]">Checking Backend...</span>
          )}
        </div>
      </div>

      {/* Top Section Grid: Specimen Input (Left) & Model Selection (Right) */}
      <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-2 items-start">
        {/* Left Column: Specimen Input, Options, and Diagnosis Button */}
        <div className="space-y-4">
          <ImageUploader
            selectedImage={selectedImage}
            previewUrl={previewUrl}
            onImageSelected={handleImageSelected}
            onClearImage={handleClearImage}
            disabled={isLoading}
          />

          {/* Explainability Toggle */}
          <div className="flex items-center justify-between rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/60 p-4 shadow-sm">
            <div>
              <span className="block text-xs font-semibold text-zinc-900 dark:text-zinc-200">
                Diagnostic Explainability Pipeline
              </span>
              <span className="text-[11px] text-zinc-500 dark:text-zinc-400">
                Compute Grad-CAM saliency & attention gating weights
              </span>
            </div>
            <input
              type="checkbox"
              id="explain-toggle"
              checked={explainEnabled}
              onChange={(e) => setExplainEnabled(e.target.checked)}
              className="h-4 w-4 rounded border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 text-emerald-600 focus:ring-emerald-500 cursor-pointer"
            />
          </div>

          {error && (
            <div className="flex items-start gap-2 rounded-xl border border-red-200 dark:border-red-900/50 bg-red-50 dark:bg-red-950/30 p-3.5 text-xs text-red-700 dark:text-red-300">
              <AlertCircle className="h-4 w-4 shrink-0 text-red-500 mt-0.5" />
              <div className="space-y-1">
                <span className="font-semibold block">Diagnosis Error</span>
                <span className="text-zinc-600 dark:text-zinc-400">{error}</span>
              </div>
            </div>
          )}

          <button
            type="button"
            onClick={handlePredict}
            disabled={isLoading || !selectedImage}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-emerald-600 dark:bg-emerald-500 py-3.5 text-sm font-semibold text-white dark:text-zinc-950 transition hover:bg-emerald-700 dark:hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-50 shadow-md shadow-emerald-950/20"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Running Checkpoint Inference...</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4 fill-current" />
                <span>Execute Leaf Diagnosis</span>
              </>
            )}
          </button>
        </div>

        {/* Right Column: Select Research Architecture */}
        <div>
          <ModelSelector
            models={RESEARCH_MODELS}
            selectedModelId={selectedModelId}
            onSelectModel={(id) => {
              setSelectedModelId(id);
              if (result && result.model_id !== id) {
                setResult(null);
              }
            }}
            disabled={isLoading}
          />
        </div>
      </div>

      {/* Bottom Section: Prediction Results & Visualizations (Underneath both input and model selector) */}
      <div className="mt-8 border-t border-zinc-200 dark:border-zinc-800/80 pt-8">
        {result ? (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
                  Diagnostic Results & Interpretability Report
                </h2>
                <p className="text-xs text-zinc-500 dark:text-zinc-400">
                  Checkpoint inference classification, confidence distribution, and visual explanations.
                </p>
              </div>
            </div>

            {/* Upper Results Row: Prediction Card & Probability Chart */}
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <PredictionCard result={result} />
              <ProbabilityChart probabilities={result.probabilities} />
            </div>

            {/* Lower Results Row: Grad-CAM Viewer & Attention Weights */}
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <GradCAMViewer
                gradcamBase64={result.gradcam_image_base64}
                gradcamView={result.gradcam_view}
                originalPreviewUrl={previewUrl}
                modelId={selectedModelId}
              />
              <AttentionWeights
                attentionData={result.attention_weights || result.attention_data}
                modelId={selectedModelId}
              />
            </div>
          </div>
        ) : (
          <div className="flex min-h-[220px] flex-col items-center justify-center rounded-xl border border-dashed border-zinc-300 dark:border-zinc-800 bg-white/40 dark:bg-zinc-900/20 p-8 text-center">
            <div className="h-12 w-12 rounded-full bg-zinc-100 dark:bg-zinc-800/80 flex items-center justify-center text-zinc-400 dark:text-zinc-500 mb-3 shadow-sm">
              <Play className="h-6 w-6 ml-0.5 text-zinc-500 dark:text-zinc-400" />
            </div>
            <p className="text-sm font-semibold text-zinc-800 dark:text-zinc-300">
              Specimen Awaiting Diagnosis
            </p>
            <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-500 max-w-md">
              Select or upload a potato leaf specimen on the left, choose a research architecture on the right, and click &quot;Execute Leaf Diagnosis&quot; to inspect real-time disease classification, softmax distributions, and diagnostic saliency below.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
