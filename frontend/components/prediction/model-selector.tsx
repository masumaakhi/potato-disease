"use client";

import React from "react";
import { ModelInfo } from "@/types/prediction";
import { Cpu, Sparkles, CheckCircle2 } from "lucide-react";

interface ModelSelectorProps {
  models: ModelInfo[];
  selectedModelId: string;
  onSelectModel: (id: string) => void;
  disabled?: boolean;
}

export function ModelSelector({
  models,
  selectedModelId,
  onSelectModel,
  disabled = false,
}: ModelSelectorProps) {
  return (
    <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/60 p-4 sm:p-5 shadow-sm transition-colors">
      <div className="flex items-center justify-between border-b border-zinc-200 dark:border-zinc-800/80 pb-3">
        <div className="flex items-center gap-2">
          <Cpu className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
            Select Research Architecture
          </h3>
        </div>
        <span className="text-[11px] font-mono text-zinc-500 dark:text-zinc-400">
          4 Models
        </span>
      </div>

      <div className="mt-4 space-y-2.5">
        {models.map((m) => {
          const isSelected = m.id === selectedModelId;
          const isProposed = m.id === "attention_fusion";

          return (
            <button
              key={m.id}
              type="button"
              disabled={disabled}
              onClick={() => onSelectModel(m.id)}
              className={`w-full rounded-lg border p-3 text-left transition ${
                isSelected
                  ? "border-emerald-500 bg-emerald-50/70 dark:bg-emerald-950/20 ring-1 ring-emerald-500/40"
                  : "border-zinc-200 dark:border-zinc-800 bg-zinc-50/60 dark:bg-zinc-950/50 hover:border-zinc-300 dark:hover:border-zinc-700 hover:bg-white dark:hover:bg-zinc-900/40"
              } ${disabled ? "cursor-not-allowed opacity-60" : ""}`}
            >
              <div className="flex flex-wrap items-center justify-between gap-1.5">
                <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
                  <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                    {m.name}
                  </span>
                  {isProposed && (
                    <span className="inline-flex items-center gap-1 rounded bg-emerald-500/10 px-1.5 py-0.5 text-[9px] sm:text-[10px] font-medium text-emerald-700 dark:text-emerald-400 border border-emerald-500/20">
                      <Sparkles className="h-2.5 w-2.5" />
                      Proposed
                    </span>
                  )}
                </div>
                {m.parameters && (
                  <span className="text-[10px] font-mono text-zinc-500 dark:text-zinc-400">
                    {m.parameters}
                  </span>
                )}
              </div>

              <div className="mt-1.5 text-[11px] text-zinc-600 dark:text-zinc-400 leading-snug">
                {m.description}
              </div>

              <div className="mt-2.5 flex items-center justify-between text-[10px] text-zinc-500 border-t border-zinc-200/50 dark:border-zinc-800/40 pt-2">
                <span className="font-mono truncate max-w-[170px] sm:max-w-none">{m.architecture}</span>
                <span className="inline-flex items-center gap-1 text-emerald-600 dark:text-emerald-400/90 font-medium shrink-0">
                  <CheckCircle2 className="h-3 w-3" />
                  Checkpoint Ready
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
