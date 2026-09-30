"use client";

import React from "react";
import { Sparkles, Layers, Eye, Info } from "lucide-react";
import { AttentionWeightsData } from "@/types/prediction";

interface AttentionWeightsProps {
  attentionData?: AttentionWeightsData | Record<string, unknown> | null;
  modelId?: string;
}

export function AttentionWeights({ attentionData, modelId }: AttentionWeightsProps) {
  const isAttentionModel = modelId === "attention_fusion";

  // Parse attention weights safely
  const cnnWeight =
    attentionData && "cnn_branch_weight" in attentionData
      ? Number(attentionData.cnn_branch_weight)
      : null;

  const vitWeight =
    attentionData && "vit_branch_weight" in attentionData
      ? Number(attentionData.vit_branch_weight)
      : null;

  const dominant =
    attentionData && "dominant_branch" in attentionData
      ? String(attentionData.dominant_branch)
      : null;

  const cnnPct = cnnWeight !== null ? (cnnWeight * 100).toFixed(1) : null;
  const vitPct = vitWeight !== null ? (vitWeight * 100).toFixed(1) : null;

  return (
    <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/60 p-4 sm:p-6 shadow-sm transition-colors">
      <div className="flex items-center justify-between border-b border-zinc-200 dark:border-zinc-800/80 pb-3">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
            Learned Branch Attention Weights
          </h3>
        </div>
        <span className="text-[10px] font-mono rounded bg-zinc-100 dark:bg-zinc-800 px-2 py-0.5 text-zinc-600 dark:text-zinc-400 border border-zinc-200 dark:border-zinc-700/60">
          α_cnn + α_vit = 1.0
        </span>
      </div>

      <p className="mt-2.5 text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
        Dynamic sample-wise softmax gating learned by the cross-attention network to balance
        inductive local CNN spatial features with global ViT semantic tokens.
      </p>

      {cnnWeight !== null && vitWeight !== null ? (
        <div className="mt-5 space-y-4">
          {/* Dual Bar Graphic */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-emerald-700 dark:text-emerald-400 font-semibold flex items-center gap-1.5">
                <Layers className="h-3.5 w-3.5" />
                CNN Branch: {cnnPct}%
              </span>
              <span className="text-teal-700 dark:text-teal-400 font-semibold flex items-center gap-1.5">
                <Eye className="h-3.5 w-3.5" />
                ViT Branch: {vitPct}%
              </span>
            </div>

            <div className="h-3 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800 flex shadow-inner">
              <div
                className="h-full bg-emerald-500 transition-all duration-700"
                style={{ width: `${cnnPct}%` }}
                title={`CNN Local Feature Weight: ${cnnPct}%`}
              />
              <div
                className="h-full bg-teal-500 dark:bg-teal-400 transition-all duration-700"
                style={{ width: `${vitPct}%` }}
                title={`ViT Global Context Weight: ${vitPct}%`}
              />
            </div>
          </div>

          {/* Dominant Branch Interpretation Badge */}
          <div className="rounded-lg border border-zinc-200 dark:border-zinc-800/80 bg-zinc-50 dark:bg-zinc-950/50 p-3">
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-zinc-500 dark:text-zinc-400 font-medium">Dominant Decision Branch:</span>
              <span className="rounded bg-emerald-500/10 px-2 py-0.5 text-xs font-bold text-emerald-700 dark:text-emerald-400 border border-emerald-500/20">
                {dominant} Branch
              </span>
            </div>
            <p className="mt-2 text-[11px] text-zinc-600 dark:text-zinc-400">
              {dominant === "CNN"
                ? "The network prioritized high-frequency localized lesion texture, necrotic rings, and edge contrast."
                : "The network prioritized broad foliar context, global leaf shape, and background noise suppression."}
            </p>
          </div>

          <div className="text-[10px] text-zinc-500 dark:text-zinc-400 flex items-start gap-1 font-mono">
            <Info className="h-3 w-3 shrink-0 mt-0.5 text-zinc-400 dark:text-zinc-500" />
            <span>Mathematical formulation: z_fused = α_cnn · z_cnn + α_vit · z_vit</span>
          </div>
        </div>
      ) : (
        <div className="mt-5 flex min-h-[140px] flex-col items-center justify-center rounded-lg border border-dashed border-zinc-300 dark:border-zinc-800/80 bg-zinc-50 dark:bg-zinc-950/40 p-5 text-center">
          <Layers className="h-8 w-8 text-zinc-400 dark:text-zinc-600 mb-2" />
          <p className="text-xs text-zinc-700 dark:text-zinc-400 font-medium">
            {isAttentionModel
              ? "Run prediction on an image to view real-time learned attention distribution."
              : "Attention weights are unique to the Attention-Guided CNN–ViT model."}
          </p>
          {!isAttentionModel && (
            <p className="mt-1 text-[11px] text-zinc-500 dark:text-zinc-500">
              Select <strong>Attention-Guided CNN–ViT</strong> from the model list above to activate dynamic branch gating.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
