import React from "react";
import { UploadCloud, Network, Eye, Award } from "lucide-react";

export function ResearchWorkflow() {
  const steps = [
    {
      icon: UploadCloud,
      step: "01",
      title: "Input Acquisition & Normalization",
      desc: "Preprocessed with research protocol (Resize 256×256 → CenterCrop 224×224 → ImageNet normalization).",
    },
    {
      icon: Network,
      step: "02",
      title: "Dual-Branch Feature Extraction",
      desc: "Concurrent execution through EfficientNet-B0 (1280d spatial features) and DeiT-Tiny (192d patch tokens).",
    },
    {
      icon: Eye,
      step: "03",
      title: "Cross-Attention Gating",
      desc: "Gating network computes sample-adaptive normalized weights [α_cnn, α_vit] to prioritize informative signals.",
    },
    {
      icon: Award,
      step: "04",
      title: "Diagnosis & Diagnostic Saliency",
      desc: "Generates disease probabilities, CNN-branch Grad-CAM diagnostic view, and quantitative branch allocation.",
    },
  ];

  return (
    <section className="py-14 border-t border-zinc-200 dark:border-zinc-900 bg-white/50 dark:bg-zinc-950/40 transition-colors">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-2xl mx-auto">
          <span className="text-xs font-mono uppercase tracking-wider text-emerald-600 dark:text-emerald-400 font-semibold">
            Execution Flow
          </span>
          <h2 className="mt-2 text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100 sm:text-3xl">
            End-to-End Pipeline & Explainability Architecture
          </h2>
          <p className="mt-2 text-xs text-zinc-600 dark:text-zinc-400">
            A unified processing chain connecting raw RGB foliage acquisition to calibrated disease classification.
          </p>
        </div>

        <div className="mt-10 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {steps.map((item) => {
            const Icon = item.icon;
            return (
              <div
                key={item.step}
                className="relative rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/60 p-6 flex flex-col justify-between transition hover:border-zinc-300 dark:hover:border-zinc-700 shadow-sm"
              >
                <div>
                  <div className="flex items-center justify-between">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                      <Icon className="h-5 w-5" />
                    </div>
                    <span className="text-xl font-bold font-mono text-zinc-400 dark:text-zinc-600">{item.step}</span>
                  </div>
                  <h3 className="mt-4 text-sm font-bold text-zinc-900 dark:text-zinc-100">{item.title}</h3>
                  <p className="mt-2 text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">{item.desc}</p>
                </div>

                <div className="mt-6 pt-3 border-t border-zinc-100 dark:border-zinc-800/60 text-[10px] text-zinc-400 dark:text-zinc-500 font-mono">
                  Stage {item.step} / 04
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
