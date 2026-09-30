"use client";

import React, { useState } from "react";
import { Table, CheckCircle2 } from "lucide-react";
import { INTERNAL_TEST_RESULTS, EXTERNAL_TEST_RESULTS } from "@/data/results";

export function ResultsTable() {
  const [activeDataset, setActiveDataset] = useState<"internal" | "external">("external");

  const results =
    activeDataset === "internal" ? INTERNAL_TEST_RESULTS : EXTERNAL_TEST_RESULTS;

  return (
    <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/60 p-4 sm:p-6 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-200 dark:border-zinc-800/80 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <Table className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
            <h3 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">
              Comparative Statistical Evaluation Table
            </h3>
          </div>
          <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-400">
            Averaged across 3 random seeds (42, 123, 456) with standard deviations.
          </p>
        </div>

        {/* Dataset selector */}
        <div className="flex items-center rounded-lg bg-zinc-100 dark:bg-zinc-950 p-1 border border-zinc-200 dark:border-zinc-800 text-xs">
          <button
            type="button"
            onClick={() => setActiveDataset("external")}
            className={`rounded-md px-3 py-1 font-medium transition ${
              activeDataset === "external"
                ? "bg-white dark:bg-zinc-800 text-teal-600 dark:text-teal-400 shadow-sm font-semibold"
                : "text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200"
            }`}
          >
            Field (PLDD-UP)
          </button>
          <button
            type="button"
            onClick={() => setActiveDataset("internal")}
            className={`rounded-md px-3 py-1 font-medium transition ${
              activeDataset === "internal"
                ? "bg-white dark:bg-zinc-800 text-emerald-600 dark:text-emerald-400 shadow-sm font-semibold"
                : "text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200"
            }`}
          >
            Lab (PlantVillage)
          </button>
        </div>
      </div>

      <div className="mt-4 overflow-x-auto -mx-1 px-1">
        <table className="w-full min-w-[500px] text-left text-xs">
          <thead>
            <tr className="border-b border-zinc-200 dark:border-zinc-800 text-[11px] font-mono uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
              <th className="py-3 pr-4 font-semibold">Model</th>
              <th className="py-3 px-3 font-semibold text-right">Accuracy</th>
              <th className="py-3 px-3 font-semibold text-right">Precision</th>
              <th className="py-3 px-3 font-semibold text-right">Recall</th>
              <th className="py-3 pl-3 font-semibold text-right text-emerald-600 dark:text-emerald-400">
                Macro F1
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800/60 font-mono">
            {results.map((r) => {
              const isProposed = r.model.includes("Proposed");

              return (
                <tr
                  key={r.model}
                  className={`transition hover:bg-zinc-50 dark:hover:bg-zinc-800/30 ${
                    isProposed ? "bg-emerald-50/60 dark:bg-emerald-950/20 font-semibold" : ""
                  }`}
                >
                  <td className="py-3 pr-4 font-sans text-zinc-800 dark:text-zinc-200 text-xs flex items-center gap-1.5">
                    {r.model}
                    {isProposed && (
                      <CheckCircle2 className="h-3 w-3 text-emerald-600 dark:text-emerald-400 shrink-0" />
                    )}
                  </td>
                  <td className="py-3 px-3 text-right text-zinc-700 dark:text-zinc-300">
                    {(r.accuracy * 100).toFixed(2)}%
                    {r.accuracy_std !== undefined && r.accuracy_std > 0 && (
                      <span className="text-[10px] text-zinc-400 dark:text-zinc-500 ml-1">
                        ±{(r.accuracy_std * 100).toFixed(1)}
                      </span>
                    )}
                  </td>
                  <td className="py-3 px-3 text-right text-zinc-700 dark:text-zinc-300">
                    {(r.precision * 100).toFixed(2)}%
                  </td>
                  <td className="py-3 px-3 text-right text-zinc-700 dark:text-zinc-300">
                    {(r.recall * 100).toFixed(2)}%
                  </td>
                  <td className="py-3 pl-3 text-right text-emerald-600 dark:text-emerald-400 font-bold">
                    {(r.f1_score * 100).toFixed(2)}%
                    {r.f1_score_std !== undefined && r.f1_score_std > 0 && (
                      <span className="text-[10px] text-emerald-600/80 dark:text-emerald-500/80 ml-1 font-normal">
                        ±{(r.f1_score_std * 100).toFixed(1)}
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="mt-4 rounded-lg bg-zinc-50 dark:bg-zinc-950/60 p-3 border border-zinc-200 dark:border-zinc-800/60 text-[11px] text-zinc-600 dark:text-zinc-400 flex items-start gap-2">
        <span className="font-semibold text-zinc-800 dark:text-zinc-300">Key Finding:</span>
        <span>
          While all models achieve ~100% accuracy in pristine laboratory conditions,
          the DeiT-Tiny Vision Transformer and attention-fused networks maintain superior
          context retention in natural fields compared to standalone CNNs.
        </span>
      </div>
    </div>
  );
}
