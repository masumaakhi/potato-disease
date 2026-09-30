"use client";

import React from "react";
import { Award, Zap, TrendingDown, Cpu } from "lucide-react";

export function MetricsCards() {
  const cards = [
    {
      label: "Controlled Lab Top Accuracy",
      icon: Award,
      value: "100.00%",
      subvalue: "PlantVillage Benchmark",
      note: "All baseline & fusion models achieve saturation under uniform laboratory illumination.",
      badge: "Laboratory Ideal",
      badgeColor: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/20",
    },
    {
      label: "Natural Field Macro F1",
      icon: Cpu,
      value: "62.38%",
      subvalue: "PLDD-UP Wild Field",
      note: "Top out-of-distribution performance in unconstrained field noise, shadows, and weeds.",
      badge: "In-The-Wild",
      badgeColor: "bg-teal-500/10 text-teal-700 dark:text-teal-400 border-teal-500/20",
    },
    {
      label: "Average Generalization Drop",
      icon: TrendingDown,
      value: "-43.2%",
      subvalue: "Domain-Shift Gap",
      note: "Substantial performance degradation when transitioning from clean lab leaves to real crop fields.",
      badge: "Research Challenge",
      badgeColor: "bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/20",
    },
    {
      label: "Proposed Model Size",
      icon: Zap,
      value: "10.11M",
      subvalue: "Total Parameters",
      note: "Lightweight footprint (38.7 MB) with 13.59 ms inference latency, suitable for edge deployment.",
      badge: "Lightweight",
      badgeColor: "bg-blue-500/10 text-blue-700 dark:text-blue-400 border-blue-500/20",
    },
  ];

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {cards.map((c) => {
        const Icon = c.icon;
        return (
          <div
            key={c.label}
            className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/60 p-5 shadow-sm transition hover:border-zinc-300 dark:hover:border-zinc-700/80"
          >
            <div className="flex items-center justify-between text-zinc-500 dark:text-zinc-400">
              <span className="text-xs font-semibold uppercase tracking-wider text-zinc-600 dark:text-zinc-400">
                {c.label}
              </span>
              <Icon className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
            </div>

            <div className="mt-3 flex items-baseline gap-2">
              <div className="text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100 font-mono">
                {c.value}
              </div>
              <span className="text-[11px] text-zinc-500 font-medium">
                {c.subvalue}
              </span>
            </div>

            <p className="mt-2 text-[11px] text-zinc-600 dark:text-zinc-400 leading-relaxed">
              {c.note}
            </p>

            <div className="mt-3 border-t border-zinc-100 dark:border-zinc-800/80 pt-2.5">
              <span
                className={`inline-block rounded border px-2 py-0.5 text-[10px] font-semibold ${c.badgeColor}`}
              >
                {c.badge}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
