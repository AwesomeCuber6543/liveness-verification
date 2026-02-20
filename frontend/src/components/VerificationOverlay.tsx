"use client";
import type { VerificationStep, OcclusionResult } from "@/lib/types";

interface VerificationOverlayProps {
  step: VerificationStep;
  faceDetected: boolean;
  occlusion: OcclusionResult | null;
}

export function VerificationOverlay({ step, faceDetected, occlusion }: VerificationOverlayProps) {
  if (step === "RESULT" || step === "CAPTURING" || step === "VERIFYING" || step === "CHALLENGE") return null;

  const isOccluded = occlusion?.is_occluded ?? false;

  // Occluded face — show warning
  if (faceDetected && isOccluded) {
    return (
      <div className="flex items-center gap-3 rounded-xl bg-amber-50 border border-amber-200 px-5 py-3 dark:bg-amber-950/30 dark:border-amber-800">
        <svg className="h-5 w-5 shrink-0 text-amber-500" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
        </svg>
        <p className="text-sm font-medium text-amber-700 dark:text-amber-300">
          Something is covering your face — please remove it
        </p>
      </div>
    );
  }

  // No face yet
  if (!faceDetected) {
    return (
      <div className="flex items-center justify-center gap-3 py-2">
        <span className="relative flex h-2.5 w-2.5 shrink-0">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-zinc-400 opacity-75" />
          <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-zinc-400" />
        </span>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Position your face in the oval
        </p>
      </div>
    );
  }

  // Face detected, clear — show hold still immediately
  return (
    <div className="flex items-center justify-center gap-3 py-2">
      <span className="relative flex h-2.5 w-2.5 shrink-0">
        <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-emerald-500" />
      </span>
      <p className="text-sm text-emerald-600 dark:text-emerald-400 font-medium">
        Hold still...
      </p>
    </div>
  );
}
