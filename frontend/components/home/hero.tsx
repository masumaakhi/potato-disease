import React from "react";
import Link from "next/link";
import { ArrowRight, Sparkles, Activity, FileText, UploadCloud } from "lucide-react";

export function Hero() {
  return (
    <section className="relative overflow-hidden py-16 sm:py-24 border-b border-zinc-200 dark:border-zinc-900 transition-colors">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          {/* Research Tag */}
          <div className="inline-flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3.5 py-1 text-xs font-semibold text-emerald-700 dark:text-emerald-400">
            <Sparkles className="h-3.5 w-3.5" />
            <span>Academic Research Demonstration</span>
          </div>

          {/* Paper Title */}
          <h1 className="mt-6 text-3xl font-extrabold tracking-tight text-zinc-900 dark:text-zinc-100 sm:text-5xl lg:text-6xl sm:leading-tight">
            An Attention-Guided Lightweight <br />
            <span className="bg-gradient-to-r from-emerald-600 via-teal-500 to-emerald-600 dark:from-emerald-400 dark:via-teal-300 dark:to-emerald-500 bg-clip-text text-transparent">
              CNN–ViT Fusion Network
            </span>
          </h1>

          <p className="mx-auto mt-4 text-sm sm:text-base font-semibold text-emerald-700 dark:text-emerald-400 max-w-2xl">
            For Potato Leaf Disease Classification Across Controlled and Natural Environments
          </p>

          <p className="mx-auto mt-4 max-w-3xl text-sm sm:text-base text-zinc-600 dark:text-zinc-400 leading-relaxed">
            Addressing the critical agricultural domain-shift challenge between pristine laboratory
            imagery (<strong>PlantVillage</strong>) and complex, in-the-wild natural field foliage
            (<strong>PLDD-UP</strong>) through dynamic local-global cross-attention gating.
          </p>

          {/* Action CTAs */}
          <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
            <Link
              href="/predict"
              className="inline-flex items-center gap-2 rounded-xl bg-emerald-600 dark:bg-emerald-500 px-6 py-3 text-sm font-semibold text-white dark:text-zinc-950 transition hover:bg-emerald-700 dark:hover:bg-emerald-400 shadow-lg shadow-emerald-950/20"
            >
              <UploadCloud className="h-4 w-4" />
              <span>Upload & Diagnose Leaf</span>
              <ArrowRight className="h-4 w-4" />
            </Link>

            <Link
              href="/results"
              className="inline-flex items-center gap-2 rounded-xl border border-zinc-300 dark:border-zinc-800 bg-white dark:bg-zinc-900/80 px-6 py-3 text-sm font-semibold text-zinc-800 dark:text-zinc-200 transition hover:bg-zinc-50 dark:hover:bg-zinc-800 hover:border-zinc-400 dark:hover:border-zinc-700 shadow-sm"
            >
              <FileText className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
              <span>Inspect Paper Benchmarks</span>
            </Link>
          </div>

          {/* Key Metrics Quick Ribbon */}
          <div className="mt-12 grid grid-cols-2 sm:grid-cols-4 gap-3 max-w-4xl mx-auto pt-8 border-t border-zinc-200 dark:border-zinc-900 text-left">
            <div className="rounded-lg bg-white dark:bg-zinc-900/40 border border-zinc-200 dark:border-zinc-800/60 p-3.5 shadow-sm">
              <span className="block text-[10px] text-zinc-500 dark:text-zinc-500 uppercase tracking-wider font-mono">
                Controlled Accuracy
              </span>
              <span className="text-lg font-bold text-zinc-900 dark:text-zinc-100 font-mono">100.0%</span>
              <span className="block text-[10px] text-emerald-600 dark:text-emerald-400 font-medium mt-0.5">PlantVillage Lab</span>
            </div>

            <div className="rounded-lg bg-white dark:bg-zinc-900/40 border border-zinc-200 dark:border-zinc-800/60 p-3.5 shadow-sm">
              <span className="block text-[10px] text-zinc-500 dark:text-zinc-500 uppercase tracking-wider font-mono">
                Field Macro F1
              </span>
              <span className="text-lg font-bold text-zinc-900 dark:text-zinc-100 font-mono">62.38%</span>
              <span className="block text-[10px] text-teal-600 dark:text-teal-400 font-medium mt-0.5">PLDD-UP Wild Field</span>
            </div>

            <div className="rounded-lg bg-white dark:bg-zinc-900/40 border border-zinc-200 dark:border-zinc-800/60 p-3.5 shadow-sm">
              <span className="block text-[10px] text-zinc-500 dark:text-zinc-500 uppercase tracking-wider font-mono">
                Proposed Params
              </span>
              <span className="text-lg font-bold text-zinc-900 dark:text-zinc-100 font-mono">10.11M</span>
              <span className="block text-[10px] text-zinc-500 dark:text-zinc-400 mt-0.5">38.7 MB Checkpoint</span>
            </div>

            <div className="rounded-lg bg-white dark:bg-zinc-900/40 border border-zinc-200 dark:border-zinc-800/60 p-3.5 shadow-sm">
              <span className="block text-[10px] text-zinc-500 dark:text-zinc-500 uppercase tracking-wider font-mono">
                Comparative Models
              </span>
              <span className="text-lg font-bold text-zinc-900 dark:text-zinc-100 font-mono">4 Trained</span>
              <span className="block text-[10px] text-zinc-500 dark:text-zinc-400 mt-0.5">CNN, ViT & Hybrids</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
