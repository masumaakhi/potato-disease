"use client";

import React, { useState } from "react";
import { Eye, AlertCircle, Layers, Image as ImageIcon, Flame } from "lucide-react";
import { GradCAMViewData } from "@/types/prediction";

interface GradCAMViewerProps {
  gradcamBase64?: string | null;
  gradcamView?: GradCAMViewData | null;
  originalPreviewUrl?: string | null;
  modelId?: string;
}

export function GradCAMViewer({
  gradcamBase64,
  gradcamView,
  originalPreviewUrl,
  modelId,
}: GradCAMViewerProps) {
  const [viewMode, setViewMode] = useState<"overlay" | "heatmap" | "original">("overlay");

  const isDeiTTiny = modelId === "deit_tiny";
  const overlayB64 = gradcamView?.gradcam_image_base64 || gradcamBase64;
  const heatmapB64 = gradcamView?.heatmap_image_base64;

  return (
    <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/60 p-4 sm:p-6 shadow-sm transition-colors">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-200 dark:border-zinc-800/80 pb-3">
        <div className="flex items-center gap-2">
          <Eye className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
            Grad-CAM — CNN Branch Diagnostic View
          </h3>
        </div>
        {gradcamView?.target_layer && (
          <span className="text-[10px] font-mono rounded bg-zinc-100 dark:bg-zinc-800 px-2 py-0.5 text-zinc-600 dark:text-zinc-400 border border-zinc-200 dark:border-zinc-700/60">
            Layer: {gradcamView.target_layer.split(" ")[0]}
          </span>
        )}
      </div>

      {/* Mandatory Research Methodological Disclaimer */}
      <div className="mt-3 flex items-start gap-2 rounded-lg border border-amber-300 dark:border-amber-900/40 bg-amber-50 dark:bg-amber-950/20 p-2.5 text-[11px] text-amber-800 dark:text-amber-300/90 leading-tight">
        <AlertCircle className="h-4 w-4 shrink-0 text-amber-600 dark:text-amber-400 mt-0.5" />
        <span>
          <strong>CNN Diagnostic Scope:</strong> Diagnostic visualization of the CNN branch only.
          This does not represent an explanation of the entire CNN–ViT hybrid model or the
          Vision Transformer self-attention branch.
        </span>
      </div>

      {/* View Mode Toggle */}
      {overlayB64 && !isDeiTTiny && (
        <div className="mt-4 flex flex-wrap items-center justify-center gap-1 rounded-lg bg-zinc-100 dark:bg-zinc-950 p-1 border border-zinc-200 dark:border-zinc-800/80 text-xs">
          <button
            type="button"
            onClick={() => setViewMode("overlay")}
            className={`flex items-center gap-1.5 rounded-md px-2.5 sm:px-3 py-1 text-[11px] sm:text-xs font-medium transition ${
              viewMode === "overlay"
                ? "bg-white dark:bg-zinc-800 text-emerald-700 dark:text-emerald-400 shadow-sm"
                : "text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200"
            }`}
          >
            <Layers className="h-3 w-3" />
            <span>Overlay View</span>
          </button>

          {heatmapB64 && (
            <button
              type="button"
              onClick={() => setViewMode("heatmap")}
              className={`flex items-center gap-1.5 rounded-md px-2.5 sm:px-3 py-1 text-[11px] sm:text-xs font-medium transition ${
                viewMode === "heatmap"
                  ? "bg-white dark:bg-zinc-800 text-emerald-700 dark:text-emerald-400 shadow-sm"
                : "text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200"
              }`}
            >
              <Flame className="h-3 w-3" />
              <span>Jet Heatmap</span>
            </button>
          )}

          {originalPreviewUrl && (
            <button
              type="button"
              onClick={() => setViewMode("original")}
              className={`flex items-center gap-1.5 rounded-md px-2.5 sm:px-3 py-1 text-[11px] sm:text-xs font-medium transition ${
                viewMode === "original"
                  ? "bg-white dark:bg-zinc-800 text-emerald-700 dark:text-emerald-400 shadow-sm"
                  : "text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200"
              }`}
            >
              <ImageIcon className="h-3 w-3" />
              <span>Original Leaf</span>
            </button>
          )}
        </div>
      )}

      {/* Display Area */}
      <div className="mt-4 flex min-h-[220px] items-center justify-center rounded-lg border border-dashed border-zinc-300 dark:border-zinc-800/80 bg-zinc-50 dark:bg-zinc-950/50 p-4 text-center">
        {isDeiTTiny ? (
          <div className="p-4 text-center max-w-sm">
            <Layers className="mx-auto h-8 w-8 text-zinc-400 dark:text-zinc-600 mb-2" />
            <p className="text-xs font-semibold text-zinc-800 dark:text-zinc-300">
              CNN Grad-CAM Not Applicable
            </p>
            <p className="mt-1 text-[11px] text-zinc-500 dark:text-zinc-400 leading-relaxed">
              DeiT-Tiny is a pure Vision Transformer and has no convolutional layers.
              Grad-CAM is strictly designed for local spatial convolutional feature maps.
            </p>
          </div>
        ) : overlayB64 ? (
          <div className="flex flex-col items-center">
            {viewMode === "overlay" && (
              <img
                src={`data:image/jpeg;base64,${overlayB64}`}
                alt="Grad-CAM CNN Branch Diagnostic View"
                className="max-h-64 w-auto rounded-lg border border-zinc-200 dark:border-zinc-800 object-contain shadow-lg"
              />
            )}
            {viewMode === "heatmap" && heatmapB64 && (
              <img
                src={`data:image/jpeg;base64,${heatmapB64}`}
                alt="Jet Colormap Heatmap"
                className="max-h-64 w-auto rounded-lg border border-zinc-200 dark:border-zinc-800 object-contain shadow-lg"
              />
            )}
            {viewMode === "original" && originalPreviewUrl && (
              <img
                src={originalPreviewUrl}
                alt="Original Leaf Specimen"
                className="max-h-64 w-auto rounded-lg border border-zinc-200 dark:border-zinc-800 object-contain shadow-lg"
              />
            )}
            <p className="mt-2 text-[10px] text-zinc-500 dark:text-zinc-400">
              {viewMode === "overlay"
                ? "Feature activation overlay (Jet colormap, α = 0.45)"
                : viewMode === "heatmap"
                ? "Normalized convolutional saliency activation (0.0 to 1.0)"
                : "Preprocessed input specimen (224×224)"}
            </p>
          </div>
        ) : (
          <div className="p-4 text-center text-zinc-500 dark:text-zinc-400 text-xs">
            <Eye className="mx-auto h-8 w-8 text-zinc-400 dark:text-zinc-700 mb-2" />
            <span>Select a model with a CNN branch and run diagnosis with explainability enabled to inspect lesion activation.</span>
          </div>
        )}
      </div>
    </div>
  );
}
