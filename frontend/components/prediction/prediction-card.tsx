"use client";

import React from "react";
import { PredictionResult } from "@/types/prediction";
import { AlertTriangle, ShieldCheck, Clock, Cpu, Maximize2 } from "lucide-react";

interface PredictionCardProps {
  result: PredictionResult;
}

export function PredictionCard({ result }: PredictionCardProps) {
  // Determine severity and pathogen based on predicted class
  const isHealthy = result.predicted_class === "Healthy";
  const isEarlyBlight = result.predicted_class === "Early Blight";
  const isLateBlight = result.predicted_class === "Late Blight";

  let pathogen = "None (Asymptomatic)";
  let severityLabel = "None";
  let severityColor = "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/20";
  let statusIcon = ShieldCheck;

  if (isEarlyBlight) {
    pathogen = "Alternaria solani (Fungal pathogen)";
    severityLabel = "High Foliar Damage";
    severityColor = "bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/20";
    statusIcon = AlertTriangle;
  } else if (isLateBlight) {
    pathogen = "Phytophthora infestans (Oomycete pathogen)";
    severityLabel = "Critical Agronomic Threat";
    severityColor = "bg-red-500/10 text-red-700 dark:text-red-400 border-red-500/20";
    statusIcon = AlertTriangle;
  }

  const StatusIcon = statusIcon;
  const confidencePct = (result.confidence * 100).toFixed(2);

  return (
    <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/60 p-6 shadow-sm transition-colors">
      {/* Top Banner: Architecture and Status */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-200 dark:border-zinc-800 pb-4">
        <div>
          <span className="text-[11px] font-mono uppercase tracking-wider text-emerald-700 dark:text-emerald-400 font-semibold">
            {result.model_name}
          </span>
          <h2 className="mt-1 text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
            <StatusIcon className={`h-6 w-6 ${isHealthy ? "text-emerald-600 dark:text-emerald-400" : isEarlyBlight ? "text-amber-500 dark:text-amber-400" : "text-red-500 dark:text-red-400"}`} />
            {result.predicted_class}
          </h2>
        </div>

        <div className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${severityColor}`}>
          <span>{severityLabel}</span>
        </div>
      </div>

      {/* Pathogen and Diagnosis Details */}
      <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="rounded-lg border border-zinc-200 dark:border-zinc-800/80 bg-zinc-50/70 dark:bg-zinc-950/40 p-3">
          <span className="text-[11px] text-zinc-500 dark:text-zinc-400">Identified Pathogen / Agent</span>
          <p className="mt-1 text-xs font-semibold italic text-zinc-800 dark:text-zinc-200">{pathogen}</p>
        </div>

        <div className="rounded-lg border border-zinc-200 dark:border-zinc-800/80 bg-zinc-50/70 dark:bg-zinc-950/40 p-3">
          <div className="flex items-center justify-between">
            <span className="text-[11px] text-zinc-500 dark:text-zinc-400">Diagnostic Confidence</span>
            <span className="text-sm font-bold text-emerald-600 dark:text-emerald-400 font-mono">{confidencePct}%</span>
          </div>
          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
            <div
              className="h-full rounded-full bg-emerald-500 transition-all duration-500"
              style={{ width: `${confidencePct}%` }}
            />
          </div>
        </div>
      </div>

      {/* Technical Inference Telemetry */}
      <div className="mt-4 flex flex-wrap items-center gap-4 rounded-lg bg-zinc-50 dark:bg-zinc-950/60 border border-zinc-200 dark:border-zinc-800/60 px-4 py-2.5 text-[11px] text-zinc-600 dark:text-zinc-400 font-mono">
        <div className="flex items-center gap-1.5">
          <Clock className="h-3.5 w-3.5 text-zinc-400 dark:text-zinc-500" />
          <span>Latency: <strong className="text-zinc-800 dark:text-zinc-200">{result.inference_time_ms.toFixed(1)} ms</strong></span>
        </div>
        <div className="flex items-center gap-1.5">
          <Cpu className="h-3.5 w-3.5 text-zinc-400 dark:text-zinc-500" />
          <span>Device: <strong className="text-zinc-800 dark:text-zinc-200 uppercase">{result.inference_info?.device || "cpu"}</strong></span>
        </div>
        <div className="flex items-center gap-1.5">
          <Maximize2 className="h-3.5 w-3.5 text-zinc-400 dark:text-zinc-500" />
          <span>Resolution: <strong className="text-zinc-800 dark:text-zinc-200">{result.inference_info?.input_resolution || "224x224"}</strong></span>
        </div>
      </div>
    </div>
  );
}
