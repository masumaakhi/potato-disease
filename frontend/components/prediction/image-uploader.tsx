"use client";

import React, { useRef, useState } from "react";
import { Upload, Image as ImageIcon, X, Sparkles, Check } from "lucide-react";
import { SAMPLE_LEAVES, SampleLeaf } from "@/data/samples";

interface ImageUploaderProps {
  selectedImage: File | null;
  previewUrl: string | null;
  onImageSelected: (file: File) => void;
  onClearImage?: () => void;
  disabled?: boolean;
}

export function ImageUploader({
  selectedImage,
  previewUrl,
  onImageSelected,
  onClearImage,
  disabled = false,
}: ImageUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedSampleId(null);
      onImageSelected(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (!disabled && e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedSampleId(null);
      onImageSelected(e.dataTransfer.files[0]);
    }
  };

  const handleSelectSample = async (sample: SampleLeaf) => {
    if (disabled) return;
    try {
      setSelectedSampleId(sample.id);
      const res = await fetch(sample.path);
      const blob = await res.blob();
      const file = new File([blob], `${sample.id}.jpg`, { type: "image/jpeg" });
      onImageSelected(file);
    } catch (err) {
      console.error("Failed to load sample image:", err);
    }
  };

  return (
    <div className="space-y-4 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/60 p-4 sm:p-5 shadow-sm transition-colors">
      <div className="flex items-center justify-between border-b border-zinc-200 dark:border-zinc-800/80 pb-3">
        <div className="flex items-center gap-2">
          <ImageIcon className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
            Potato Leaf Specimen Input
          </h3>
        </div>
        {previewUrl && onClearImage && (
          <button
            type="button"
            onClick={onClearImage}
            disabled={disabled}
            className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-red-500 transition"
          >
            <X className="h-3 w-3" />
            <span>Clear</span>
          </button>
        )}
      </div>

      {/* Upload Zone */}
      <div
        onClick={() => !disabled && fileInputRef.current?.click()}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative flex min-h-[160px] sm:min-h-[190px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed transition ${
          isDragging
            ? "border-emerald-500 bg-emerald-50/70 dark:bg-emerald-950/20"
            : "border-zinc-300 dark:border-zinc-800 bg-zinc-50/60 dark:bg-zinc-950/40 hover:border-zinc-400 dark:hover:border-zinc-700 hover:bg-zinc-100/50 dark:hover:bg-zinc-900/40"
        } ${disabled ? "cursor-not-allowed opacity-60" : ""}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp,image/tiff"
          className="hidden"
          onChange={handleFileChange}
          disabled={disabled}
        />

        {previewUrl ? (
          <div className="relative flex flex-col items-center p-3">
            <img
              src={previewUrl}
              alt="Leaf Preview"
              className="max-h-40 w-auto rounded-lg border border-zinc-200 dark:border-zinc-800 object-contain shadow-md"
            />
            <p className="mt-2 text-[11px] text-zinc-600 dark:text-zinc-400 font-mono">
              {selectedImage ? selectedImage.name : "Loaded Specimen"}
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center p-5 text-center">
            <div className="flex h-11 w-11 items-center justify-center rounded-full bg-zinc-100 dark:bg-zinc-800/80 text-emerald-600 dark:text-emerald-400 shadow-sm">
              <Upload className="h-5 w-5" />
            </div>
            <p className="mt-2.5 text-xs font-semibold text-zinc-800 dark:text-zinc-200">
              Drop potato leaf image here or click to browse
            </p>
            <p className="mt-1 text-[11px] text-zinc-500 dark:text-zinc-400">
              Validated formats: JPEG, PNG, WEBP, BMP (Input cropped to 224×224)
            </p>
          </div>
        )}
      </div>

      {/* Quick Select Research Samples */}
      <div>
        <div className="flex items-center gap-1.5 text-[11px] font-medium text-zinc-600 dark:text-zinc-400 mb-2">
          <Sparkles className="h-3 w-3 text-emerald-600 dark:text-emerald-400" />
          <span>Or test with benchmark specimen samples:</span>
        </div>
        <div className="grid grid-cols-3 gap-2">
          {SAMPLE_LEAVES.map((sample) => {
            const isChosen = selectedSampleId === sample.id;
            return (
              <button
                key={sample.id}
                type="button"
                disabled={disabled}
                onClick={() => handleSelectSample(sample)}
                className={`flex flex-col items-center rounded-lg border p-2 text-center transition ${
                  isChosen
                    ? "border-emerald-500 bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-300 ring-1 ring-emerald-500/40"
                    : "border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950/60 text-zinc-600 dark:text-zinc-400 hover:border-zinc-300 dark:hover:border-zinc-700 hover:text-zinc-900 dark:hover:text-zinc-200"
                } ${disabled ? "cursor-not-allowed opacity-60" : ""}`}
              >
                <div className="h-10 w-10 overflow-hidden rounded border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900">
                  <img
                    src={sample.path}
                    alt={sample.name}
                    className="h-full w-full object-cover"
                  />
                </div>
                <span className="mt-1.5 text-[10px] font-medium truncate max-w-full">
                  {sample.category}
                </span>
                {isChosen && (
                  <span className="mt-0.5 inline-flex items-center text-[9px] text-emerald-600 dark:text-emerald-400 font-semibold">
                    <Check className="h-2.5 w-2.5 mr-0.5" /> Selected
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
