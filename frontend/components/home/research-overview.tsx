import React from "react";
import { RESEARCH_MODELS } from "@/data/models";

export function ResearchOverview() {
  return (
    <section className="py-14 transition-colors">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 space-y-16">
        {/* Section 1: Research Motivation & The Domain-Shift Challenge */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          <div className="lg:col-span-6 space-y-4">
            <span className="text-xs font-mono uppercase tracking-wider text-emerald-600 dark:text-emerald-400 font-semibold">
              Research Motivation
            </span>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
              The Controlled vs. Natural Field Generalization Gap
            </h2>
            <p className="text-sm text-zinc-600 dark:text-zinc-400 leading-relaxed">
              Standard deep learning benchmarks often train on laboratory datasets with isolated,
              single-leaf specimens photographed under uniform artificial lighting (<strong>PlantVillage</strong>).
              While standard CNNs and Vision Transformers achieve nearly 100% accuracy in these artificial settings,
              they experience catastrophic generalization degradation when deployed to real farm fields (<strong>PLDD-UP</strong>).
            </p>
            <p className="text-sm text-zinc-600 dark:text-zinc-400 leading-relaxed">
              Real agricultural conditions introduce complex background clutter, direct sunlight glare,
              soil reflectance, overlapping weed foliage, and multi-leaf clusters that confuse monolithic models.
            </p>
          </div>

          <div className="lg:col-span-6 grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/60 p-5 space-y-2.5 shadow-sm">
              <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400 font-mono">
                Dataset 01: PlantVillage
              </span>
              <h3 className="text-base font-bold text-zinc-900 dark:text-zinc-100">Controlled Laboratory</h3>
              <ul className="text-xs text-zinc-600 dark:text-zinc-400 space-y-1.5 list-disc pl-4">
                <li>Isolated single leaf on neutral white/gray background</li>
                <li>Uniform studio illumination</li>
                <li>High signal-to-noise ratio</li>
                <li>Baseline models achieve ~100% accuracy</li>
              </ul>
            </div>

            <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/60 p-5 space-y-2.5 shadow-sm">
              <span className="text-xs font-semibold text-teal-600 dark:text-teal-400 font-mono">
                Dataset 02: PLDD-UP
              </span>
              <h3 className="text-base font-bold text-zinc-900 dark:text-zinc-100">Natural Agricultural Field</h3>
              <ul className="text-xs text-zinc-600 dark:text-zinc-400 space-y-1.5 list-disc pl-4">
                <li>In-situ field leaves attached to live plants</li>
                <li>Variable solar angles, wind motion & shadows</li>
                <li>Background clutter: soil, mulch, adjacent weeds</li>
                <li>Induces 36% – 51% accuracy drop in baselines</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Section 2: Architectural Methodology & Synergies */}
        <div className="space-y-8">
          <div className="text-center max-w-3xl mx-auto">
            <span className="text-xs font-mono uppercase tracking-wider text-emerald-600 dark:text-emerald-400 font-semibold">
              Methodological Framework
            </span>
            <h2 className="mt-2 text-2xl sm:text-3xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
              Synergizing Inductive Bias with Global Attention
            </h2>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              The proposed architecture systematically bridges the gap between local convolutional feature
              maps and long-range self-attention token representations.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {RESEARCH_MODELS.map((m) => {
              const isProposed = m.id === "attention_fusion";

              return (
                <div
                  key={m.id}
                  className={`rounded-xl border p-6 flex flex-col justify-between transition shadow-sm ${
                    isProposed
                      ? "border-emerald-500/50 dark:border-emerald-500/50 bg-emerald-50/50 dark:bg-emerald-950/20 ring-1 ring-emerald-500/30"
                      : "border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/60"
                  }`}
                >
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                        {m.parameters}
                      </span>
                      {isProposed && (
                        <span className="rounded bg-emerald-600/10 dark:bg-emerald-500/20 px-2 py-0.5 text-[9px] font-bold text-emerald-700 dark:text-emerald-400 uppercase tracking-wider">
                          Proposed Model
                        </span>
                      )}
                    </div>

                    <h3 className="text-base font-bold text-zinc-900 dark:text-zinc-100">{m.name}</h3>
                    <span className="text-xs font-mono text-emerald-700 dark:text-emerald-400/90 block font-medium">
                      {m.architecture}
                    </span>

                    <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                      {m.description}
                    </p>
                  </div>

                  <div className="mt-6 pt-3 border-t border-zinc-100 dark:border-zinc-800/80 text-[11px] text-zinc-500 dark:text-zinc-400 flex items-center justify-between">
                    <span>Validation Checkpoint</span>
                    <span className="text-emerald-700 dark:text-emerald-400 font-mono font-medium">Trained (Seed 42)</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Section 3: Target Foliar Pathology */}
        <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/40 p-8 space-y-6 shadow-sm">
          <div className="text-left">
            <span className="text-xs font-mono uppercase tracking-wider text-emerald-600 dark:text-emerald-400 font-semibold">
              Agronomic Classification Target
            </span>
            <h2 className="mt-1 text-xl sm:text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
              Target Potato Foliage Pathologies
            </h2>
            <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
              Trained across three distinct clinical categories defined in the research protocol.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950/60 p-5 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-zinc-500 dark:text-zinc-400">Class ID 0</span>
                <span className="rounded bg-amber-500/10 px-2 py-0.5 text-[10px] font-semibold text-amber-600 dark:text-amber-400 border border-amber-500/20">
                  High Severity
                </span>
              </div>
              <h3 className="text-base font-bold text-zinc-900 dark:text-zinc-100">Early Blight</h3>
              <p className="text-xs italic text-zinc-600 dark:text-zinc-400">Alternaria solani</p>
              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed pt-2 border-t border-zinc-200 dark:border-zinc-800/60">
                Identified by characteristic concentric dark rings (&quot;target-board&quot; pattern) surrounded by chlorotic yellow halos on mature lower leaves.
              </p>
            </div>

            <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950/60 p-5 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-zinc-500 dark:text-zinc-400">Class ID 1</span>
                <span className="rounded bg-red-500/10 px-2 py-0.5 text-[10px] font-semibold text-red-600 dark:text-red-400 border border-red-500/20">
                  Critical Threat
                </span>
              </div>
              <h3 className="text-base font-bold text-zinc-900 dark:text-zinc-100">Late Blight</h3>
              <p className="text-xs italic text-zinc-600 dark:text-zinc-400">Phytophthora infestans</p>
              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed pt-2 border-t border-zinc-200 dark:border-zinc-800/60">
                Rapidly expanding water-soaked dark brown-to-black necrotic lesions that consume entire leaf clusters during humid, cool agricultural cycles.
              </p>
            </div>

            <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950/60 p-5 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-zinc-500 dark:text-zinc-400">Class ID 2</span>
                <span className="rounded bg-emerald-500/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-700 dark:text-emerald-400 border border-emerald-500/20">
                  Asymptomatic
                </span>
              </div>
              <h3 className="text-base font-bold text-zinc-900 dark:text-zinc-100">Healthy Leaf</h3>
              <p className="text-xs italic text-zinc-600 dark:text-zinc-400">Normal Physiology</p>
              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed pt-2 border-t border-zinc-200 dark:border-zinc-800/60">
                Vibrant, uniform green chlorophyll distribution with clean leaf margins, intact vein networks, and absence of fungal or oomycete necrosis.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
