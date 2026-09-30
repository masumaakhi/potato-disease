import React from "react";
import { MetricsCards } from "@/components/results/metrics-cards";
import { ModelComparison } from "@/components/results/model-comparison";
import { EfficiencyChart } from "@/components/results/efficiency-chart";
import { ResultsTable } from "@/components/results/results-table";

export default function ResultsPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="border-b border-zinc-200 dark:border-zinc-800 pb-5">
        <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100 sm:text-3xl">
          Research Evaluation & Benchmark Metrics
        </h1>
        <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
          Empirical comparison between Lightweight CNN (EfficientNet-B0), Compact ViT (DeiT-Tiny),
          Concat Baseline, and Proposed Attention-Guided Fusion across PlantVillage and PLDD-UP datasets.
        </p>
      </div>

      <div className="mt-8 space-y-8">
        <MetricsCards />
        <ModelComparison />
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
          <EfficiencyChart />
          <ResultsTable />
        </div>
      </div>
    </div>
  );
}
