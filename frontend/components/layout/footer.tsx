import React from "react";

export function Footer() {
  return (
    <footer className="w-full border-t border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 py-8 text-zinc-500 transition-colors">
      <div className="mx-auto max-w-7xl px-4 text-center text-xs sm:px-6 lg:px-8">
        <p className="font-semibold text-zinc-700 dark:text-zinc-300">
          An Attention-Guided Lightweight CNN–ViT Fusion Network for Potato Leaf Disease Classification
        </p>
        <p className="mt-2 text-zinc-500 dark:text-zinc-500">
          Research Demonstration Platform • Controlled (PlantVillage) & Natural Environment (PLDD-UP) Evaluation
        </p>
      </div>
    </footer>
  );
}
