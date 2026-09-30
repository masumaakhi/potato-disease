"use client";

import React, { useState } from "react";
import { GitCompare } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import {
  INTERNAL_TEST_RESULTS,
  EXTERNAL_TEST_RESULTS,
  GENERALIZATION_GAP_DATA,
} from "@/data/results";

export function ModelComparison() {
  const [activeMetric, setActiveMetric] = useState<"f1" | "accuracy">("f1");

  // Prepare chart comparison data
  const chartData = INTERNAL_TEST_RESULTS.map((internal, idx) => {
    const external = EXTERNAL_TEST_RESULTS[idx];
    const shortName = internal.model
      .replace("Baseline ", "")
      .replace("Proposed", "")
      .replace("Concatenation", "Concat")
      .trim();

    return {
      model: shortName,
      internal:
        activeMetric === "f1"
          ? Number((internal.f1_score * 100).toFixed(2))
          : Number((internal.accuracy * 100).toFixed(2)),
      external:
        activeMetric === "f1"
          ? Number((external.f1_score * 100).toFixed(2))
          : Number((external.accuracy * 100).toFixed(2)),
      drop:
        activeMetric === "f1"
          ? Number(((internal.f1_score - external.f1_score) * 100).toFixed(2))
          : Number(((internal.accuracy - external.accuracy) * 100).toFixed(2)),
    };
  });

  return (
    <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/60 p-4 sm:p-6 shadow-sm transition-colors">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-zinc-200 dark:border-zinc-800/80 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <GitCompare className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
            <h3 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">
              Controlled Laboratory vs. Natural Field Generalization
            </h3>
          </div>
          <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-400">
            Empirical demonstration of the domain shift: models trained in pristine laboratory conditions
            suffer significant performance degradation under wild agricultural field noise.
          </p>
        </div>

        {/* Metric Selector Toggle */}
        <div className="flex items-center rounded-lg bg-zinc-100 dark:bg-zinc-950 p-1 border border-zinc-200 dark:border-zinc-800 text-xs">
          <button
            type="button"
            onClick={() => setActiveMetric("f1")}
            className={`rounded-md px-3 py-1 font-medium transition ${
              activeMetric === "f1"
                ? "bg-white dark:bg-zinc-800 text-emerald-700 dark:text-emerald-400 shadow-sm font-semibold"
                : "text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200"
            }`}
          >
            Macro F1 Score
          </button>
          <button
            type="button"
            onClick={() => setActiveMetric("accuracy")}
            className={`rounded-md px-3 py-1 font-medium transition ${
              activeMetric === "accuracy"
                ? "bg-white dark:bg-zinc-800 text-emerald-700 dark:text-emerald-400 shadow-sm font-semibold"
                : "text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200"
            }`}
          >
            Overall Accuracy
          </button>
        </div>
      </div>

      {/* Comparative Bar Chart */}
      <div className="mt-6 h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            margin={{ top: 20, right: 10, left: -15, bottom: 20 }}
          >
            <XAxis
              dataKey="model"
              stroke="#71717a"
              tick={{ fontSize: 11 }}
              axisLine={{ stroke: "#71717a" }}
              tickLine={{ stroke: "#71717a" }}
            />
            <YAxis
              domain={[0, 105]}
              unit="%"
              stroke="#71717a"
              tick={{ fontSize: 11 }}
              axisLine={{ stroke: "#71717a" }}
              tickLine={{ stroke: "#71717a" }}
            />
            <Tooltip
              formatter={(val: number, name: string) => [
                `${val}%`,
                name === "internal" ? "Controlled (PlantVillage)" : "Natural Field (PLDD-UP)",
              ]}
              contentStyle={{
                backgroundColor: "var(--background)",
                borderColor: "#71717a",
                borderRadius: "8px",
                fontSize: "12px",
                color: "var(--foreground)",
              }}
            />
            <Legend
              wrapperStyle={{ paddingTop: "10px", fontSize: "12px" }}
              formatter={(val) =>
                val === "internal"
                  ? "Controlled Laboratory (PlantVillage)"
                  : "Natural Field In-The-Wild (PLDD-UP)"
              }
            />
            <Bar
              dataKey="internal"
              fill="#10b981"
              radius={[4, 4, 0, 0]}
              name="internal"
            />
            <Bar
              dataKey="external"
              fill="#0d9488"
              radius={[4, 4, 0, 0]}
              name="external"
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Generalization Drop Callout Grid */}
      <div className="mt-6 grid grid-cols-2 sm:grid-cols-4 gap-3 border-t border-zinc-200 dark:border-zinc-800/80 pt-4">
        {GENERALIZATION_GAP_DATA.map((item) => (
          <div
            key={item.model}
            className="rounded-lg border border-zinc-200 dark:border-zinc-800/80 bg-zinc-50 dark:bg-zinc-950/40 p-3"
          >
            <span className="block text-[11px] font-semibold text-zinc-800 dark:text-zinc-300 truncate">
              {item.model}
            </span>
            <div className="mt-2 flex items-baseline justify-between">
              <span className="text-[10px] text-zinc-500">Generalization Gap</span>
              <span className="text-xs font-mono font-bold text-red-500 dark:text-red-400">
                -{item.macro_f1_drop_pct.toFixed(1)}%
              </span>
            </div>
            <div className="mt-1 flex items-center justify-between text-[10px] text-zinc-600 dark:text-zinc-400 font-mono">
              <span>100% → {item.external_macro_f1.toFixed(1)}%</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
