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
import { Play, AlertCircle, AlertTriangle, Wifi, WifiOff, Loader2, Check } from "lucide-react";

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
              <span className="text-red-600 dark:text-red-300 font-mono text-[11px] font-medium">Backend Offline</span>
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
            <div className="rounded-xl border border-red-200 dark:border-red-900/60 bg-red-50/95 dark:bg-red-950/40 p-4 text-xs text-red-900 dark:text-red-200 shadow-sm transition-all duration-200 space-y-3">
              <div className="flex items-start gap-3">
                <div className="rounded-full bg-red-100 dark:bg-red-900/60 p-1.5 text-red-600 dark:text-red-400 shrink-0 mt-0.5">
                  <AlertCircle className="h-4 w-4" />
                </div>
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-sm text-red-950 dark:text-red-100">
                      নমুনা বা ডায়াগনসিস সমস্যা (Specimen / Execution Notice)
                    </span>
                  </div>
                  <div className="text-red-800 dark:text-red-300/90 leading-relaxed font-sans">
                    {error.includes("NetworkError") || error.includes("Failed to fetch") ? (
                      <>
                        <p className="font-medium text-red-900 dark:text-red-200 mb-1">
                          ভুল বা অপ্রাসঙ্গিক ইমেজ সনাক্তকরণ:
                        </p>
                        <p>
                          আপনার আপলোড করা ছবিটি হয়তো একটি মোবাইল স্ক্রিনশট বা কোনো আলুর পাতার ছবি নয়। এই রিসার্চ এআই মডেলটি শুধুমাত্র আলুর পাতার ৩টি অবস্থার (Early Blight, Late Blight, Healthy) ওপর ক্যালিব্রেটেড। এছাড়াও ক্লাউড ব্যাকএন্ডটি স্লিপ থেকে সচল হতে কয়েক সেকেন্ড সময় নিতে পারে।
                        </p>
                      </>
                    ) : (
                      <p>{error}</p>
                    )}
                  </div>
                </div>
              </div>

              {/* Actionable guidance box */}
              <div className="rounded-lg border border-red-200/80 dark:border-red-900/40 bg-white/70 dark:bg-zinc-900/50 p-3 text-[11px] text-zinc-700 dark:text-zinc-300 space-y-2">
                <p className="font-semibold text-zinc-900 dark:text-zinc-200">
                  💡 আপনার ভুল কোথায় এবং কীভাবে সমাধান করবেন:
                </p>
                <ul className="list-disc pl-4 space-y-1 text-zinc-600 dark:text-zinc-400">
                  <li>
                    <strong>ছবি ক্রপ করুন:</strong> পুরো মোবাইল ডিসপ্লের স্ক্রিনশট না দিয়ে, শুধুমাত্র গাছের পাতাটির অংশ ক্রপ করে দিন।
                  </li>
                  <li>
                    <strong>আসল পাতা দিন:</strong> পাতার ওপর আলো ভালো থাকলে এবং ক্যামেরা কাছে রেখে ছবি তুললে মডেল নিখুঁত রোগ চিহ্নিত করতে পারে।
                  </li>
                  <li>
                    <strong>সিস্টেম পরীক্ষা করতে:</strong> নিচে দেওয়া ৩টি নমুনা পাতার (Early Blight / Late Blight / Healthy) যেকোনো একটিতে ক্লিক করে এখনই সিস্টেমটি টেস্ট করে দেখতে পারেন।
                  </li>
                </ul>

                <div className="flex flex-wrap items-center gap-2 pt-1">
                  <button
                    type="button"
                    onClick={async () => {
                      setError(null);
                      try {
                        const res = await fetch("/samples/sample_healthy.jpg");
                        const blob = await res.blob();
                        const file = new File([blob], "sample_healthy.jpg", { type: "image/jpeg" });
                        handleImageSelected(file);
                      } catch (e) {
                        console.error(e);
                      }
                    }}
                    className="inline-flex items-center gap-1 rounded-md bg-emerald-600 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-emerald-700 transition"
                  >
                    <Check className="h-3 w-3" />
                    <span>নমুনা পাতা দিয়ে টেস্ট করুন</span>
                  </button>
                  <button
                    type="button"
                    onClick={handleClearImage}
                    className="inline-flex items-center gap-1 rounded-md border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-2.5 py-1 text-[11px] font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-zinc-700/60 transition"
                  >
                    <span>ছবি পরিবর্তন করুন</span>
                  </button>
                </div>
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

            {/* Specimen Quality & Out-of-Distribution (OOD) Advisory Banner */}
            {result.specimen_warning && (
              <div className="flex items-start gap-3.5 rounded-xl border border-amber-300 dark:border-amber-700/60 bg-amber-50/90 dark:bg-amber-950/30 p-4 text-xs text-amber-900 dark:text-amber-200 shadow-sm transition-all duration-300">
                <AlertTriangle className="h-5 w-5 shrink-0 text-amber-600 dark:text-amber-400 mt-0.5" />
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-sm text-amber-950 dark:text-amber-100">
                      Specimen Suitability Advisory
                    </span>
                    <span className="rounded-full bg-amber-200/80 dark:bg-amber-800/60 px-2 py-0.5 text-[10px] font-semibold text-amber-800 dark:text-amber-200 uppercase tracking-wider">
                      Out of Distribution
                    </span>
                  </div>
                  <p className="text-amber-800 dark:text-amber-300/90 leading-relaxed font-sans">
                    {result.specimen_warning}
                  </p>
                </div>
              </div>
            )}

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
