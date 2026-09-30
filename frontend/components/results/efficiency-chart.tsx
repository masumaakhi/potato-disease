"use client";

import React from "react";
import { Zap, HardDrive, Cpu, Gauge } from "lucide-react";
import { EFFICIENCY_COMPARISON_DATA } from "@/data/results";

export function EfficiencyChart() {
  return (
    <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/60 p-4 sm:p-6 shadow-sm">
      <div className="flex items-center gap-2 border-b border-zinc-200 dark:border-zinc-800/80 pb-3">
        <Zap className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
        <h3 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">
          Computational Efficiency & Resource Footprint
        </h3>
      </div>
      <p className="mt-2 text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
        Benchmarked on NVIDIA Tesla T4 GPU with batch size 16 (224 × 224 input resolution).
        The proposed model balances edge portability with expressive cross-branch attention.
      </p>

      <div className="mt-5 space-y-3.5">
        {EFFICIENCY_COMPARISON_DATA.map((entry) => {
          const isProposed = entry.model.includes("Proposed");

          return (
            <div
              key={entry.model}
              className={`rounded-lg border p-3.5 sm:p-4 transition ${
                isProposed
                  ? "border-emerald-500/50 bg-emerald-50/50 dark:bg-emerald-950/20 ring-1 ring-emerald-500/20"
                  : "border-zinc-200 dark:border-zinc-800 bg-zinc-50/60 dark:bg-zinc-950/40"
              }`}
            >
              <div className="flex flex-wrap items-center justify-between gap-1.5">
                <span className="text-xs font-bold text-zinc-800 dark:text-zinc-200 flex items-center gap-2">
                  {entry.model}
                  {isProposed && (
                    <span className="rounded bg-emerald-100 dark:bg-emerald-500/20 px-1.5 py-0.5 text-[9px] font-semibold text-emerald-700 dark:text-emerald-400 uppercase tracking-wider">
                      Proposed
                    </span>
                  )}
                </span>
                <span className="text-xs font-mono font-semibold text-zinc-600 dark:text-zinc-300">
                  {entry.params_formatted}
                </span>
              </div>

              <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2.5 border-t border-zinc-200/80 dark:border-zinc-800/60 pt-2.5 text-[11px] font-mono text-zinc-500 dark:text-zinc-400">
                <div>
                  <span className="block text-[9px] uppercase tracking-wider text-zinc-400 dark:text-zinc-500 font-sans">
                    Weights
                  </span>
                  <span className="text-zinc-800 dark:text-zinc-200 font-medium">{entry.weights_mb.toFixed(1)} MB</span>
                </div>
                <div>
                  <span className="block text-[9px] uppercase tracking-wider text-zinc-400 dark:text-zinc-500 font-sans">
                    Complexity
                  </span>
                  <span className="text-zinc-800 dark:text-zinc-200 font-medium">{entry.gflops.toFixed(2)} GFLOPs</span>
                </div>
                <div>
                  <span className="block text-[9px] uppercase tracking-wider text-zinc-400 dark:text-zinc-500 font-sans">
                    Latency
                  </span>
                  <span className="text-zinc-800 dark:text-zinc-200 font-medium">{entry.latency_ms.toFixed(1)} ms</span>
                </div>
                <div>
                  <span className="block text-[9px] uppercase tracking-wider text-zinc-400 dark:text-zinc-500 font-sans">
                    Throughput
                  </span>
                  <span className="text-emerald-600 dark:text-emerald-400 font-bold">{entry.throughput_fps.toFixed(0)} fps</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
