import React from "react";
import { Hero } from "@/components/home/hero";
import { ResearchOverview } from "@/components/home/research-overview";
import { ResearchWorkflow } from "@/components/home/research-workflow";

export default function HomePage() {
  return (
    <div className="space-y-12 pb-16">
      <Hero />
      <ResearchOverview />
      <ResearchWorkflow />
    </div>
  );
}
