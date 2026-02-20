"use client";
import { RefObject } from "react";
import { FRAME_WIDTH, FRAME_HEIGHT } from "@/lib/constants";
import type { VerificationStep, OcclusionResult } from "@/lib/types";

interface CameraFeedProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  step: VerificationStep;
  faceDetected: boolean;
  occlusion: OcclusionResult | null;
}

export function CameraFeed({ videoRef, canvasRef, step, faceDetected, occlusion }: CameraFeedProps) {
  const isProcessing = step === "CAPTURING" || step === "VERIFYING";
  // Face found + not occluded but still in CHECKING_OCCLUSION (building clear streak)
  const isFaceLocked = step === "CHECKING_OCCLUSION" && faceDetected && occlusion !== null && !occlusion.is_occluded;
  const shouldBlur = isProcessing || isFaceLocked;

  return (
    <>
      <div
        style={{ width: "100%", aspectRatio: "3/4" }}
        className="relative mx-auto overflow-hidden rounded-3xl bg-black"
      >
        {/* Video */}
        <video
          ref={videoRef}
          style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)" }}
          className={`transition-all duration-500 ${
            shouldBlur ? "blur-sm brightness-50 grayscale" : ""
          }`}
          width={FRAME_WIDTH}
          height={FRAME_HEIGHT}
          playsInline
          muted
        />

        {/* SVG overlay — dark mask with ellipse cutout */}
        <svg
          style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}
          viewBox="0 0 300 400"
          preserveAspectRatio="none"
        >
          <defs>
            <mask id="oval-cutout">
              <rect width="300" height="400" fill="white" />
              <ellipse cx="150" cy="180" rx="110" ry="150" fill="black" />
            </mask>
          </defs>
          <rect
            width="300"
            height="400"
            fill="rgba(0,0,0,0.75)"
            mask="url(#oval-cutout)"
          />
          <ellipse
            cx="150"
            cy="180"
            rx="110"
            ry="150"
            fill="none"
            stroke="rgba(255,255,255,0.3)"
            strokeWidth="2"
          />
        </svg>

        {/* Loading overlay */}
        {shouldBlur && (
          <div style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16 }}>
            <div className="relative h-14 w-14">
              <div className="absolute inset-0 animate-spin rounded-full border-[3px] border-white/20 border-t-white" />
              <div
                className="absolute inset-2 animate-spin rounded-full border-[3px] border-white/10 border-b-white/60"
                style={{ animationDirection: "reverse", animationDuration: "1.5s" }}
              />
            </div>
            <p className="text-sm font-medium text-white">
              {isProcessing
                ? step === "CAPTURING" ? "Capturing..." : "Verifying..."
                : "Hold still..."
              }
            </p>
          </div>
        )}
      </div>

      <canvas ref={canvasRef} style={{ display: "none" }} />
    </>
  );
}
