"use client";

import React from "react";
import { ClassProbability } from "@/types/prediction";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { BarChart3 } from "lucide-react";

interface ProbabilityChartProps {
  probabilities: ClassProbability[];
}

export function ProbabilityChart({ probabilities }: ProbabilityChartProps) {
  // Sort descending by probability for clean visual hierarchy
  const chartData = probabilities.map((p) => ({
    name: p.class_name,
    probability: Number((p.probability * 100).toFixed(2)),
    percentage: p.percentage,
  }));

  // Identify top class
  const maxProb = Math.max(...chartData.map((d) => d.probability));

  return (
    <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/60 p-4 sm:p-6 shadow-sm transition-colors">
      <div className="flex items-center justify-between border-b border-zinc-200 dark:border-zinc-800/80 pb-3">
        <div className="flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
            Softmax Disease Probability Distribution
          </h3>
        </div>
        <span className="text-[11px] font-mono text-zinc-500 dark:text-zinc-400">
          ∑ p_i ≈ 1.0
        </span>
      </div>

      <div className="mt-4 h-48 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 20, left: -5, bottom: 5 }}
          >
            <XAxis
              type="number"
              domain={[0, 100]}
              unit="%"
              stroke="#71717a"
              tick={{ fontSize: 10 }}
              axisLine={{ stroke: "#71717a" }}
              tickLine={{ stroke: "#71717a" }}
            />
            <YAxis
              type="category"
              dataKey="name"
              stroke="#71717a"
              width={95}
              tick={{ fontSize: 11 }}
              axisLine={{ stroke: "#71717a" }}
              tickLine={false}
            />
            <Tooltip
              formatter={(val: number) => [`${val}%`, "Confidence"]}
              contentStyle={{
                backgroundColor: "var(--background)",
                borderColor: "#71717a",
                borderRadius: "8px",
                fontSize: "12px",
                color: "var(--foreground)",
              }}
            />
            <Bar dataKey="probability" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, index) => {
                const isTop = entry.probability === maxProb;
                return (
                  <Cell
                    key={`cell-${index}`}
                    fill={isTop ? "#10b981" : "#71717a"}
                  />
                );
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-3 grid grid-cols-3 gap-2 border-t border-zinc-200 dark:border-zinc-800/80 pt-3 text-center">
        {chartData.map((d) => (
          <div key={d.name} className="rounded bg-zinc-50 dark:bg-zinc-950/40 p-1.5 border border-zinc-200/60 dark:border-zinc-800/40">
            <span className="block text-[10px] text-zinc-500 dark:text-zinc-400 truncate">{d.name}</span>
            <span className="text-xs font-mono font-semibold text-zinc-800 dark:text-zinc-200">
              {d.percentage}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
