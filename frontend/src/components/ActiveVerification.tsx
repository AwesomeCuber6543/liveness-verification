"use client";
import { useEffect, useRef } from "react";
import { useVerification } from "@/hooks/useVerification";
import { CameraFeed } from "@/components/CameraFeed";
import { VerificationOverlay } from "@/components/VerificationOverlay";
import { ChallengeChecklist } from "@/components/ChallengeChecklist";

export default function ActiveVerification() {
  const v = useVerification("active");
  const startedRef = useRef(false);

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;

    console.log("[Page] Mounting active verification, starting...");
    v.start();

    return () => {
      console.log("[Page] Unmounting active verification, stopping...");
      v.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (v.step === "RESULT" && v.result) {
      v.stop();
      const params = new URLSearchParams({ passed: String(v.result.passed) });
      window.location.href = `/verify/active/result?${params.toString()}`;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [v.step, v.result]);

  const showChecklist = v.step === "CHALLENGE";

  return (
    <div className="flex min-h-screen flex-col bg-zinc-50 dark:bg-zinc-950">
      <header className="border-b border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mx-auto flex max-w-2xl items-center gap-4 px-6 py-4">
          <a
            href="/"
            className="flex items-center gap-1 text-sm text-zinc-500 transition-colors hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-50"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5 8.25 12l7.5-7.5" />
            </svg>
            Back
          </a>
          <h1 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">
            Active Verification
          </h1>
        </div>
      </header>

      <div className="flex flex-1 flex-col items-center justify-center px-6 py-10">
        <div className="w-full max-w-md flex flex-col items-center gap-5">
          {(v.cameraError || v.wsError) && (
            <div className="w-full rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700 dark:bg-red-950/30 dark:text-red-400">
              {v.cameraError || v.wsError}
            </div>
          )}

          <CameraFeed
            videoRef={v.videoRef}
            canvasRef={v.canvasRef}
            step={v.step}
            faceDetected={v.faceDetected}
            occlusion={v.occlusion}
          />

          {showChecklist ? (
            <div className="w-full">
              <ChallengeChecklist challenge={v.challenge} />
            </div>
          ) : (
            <VerificationOverlay
              step={v.step}
              faceDetected={v.faceDetected}
              occlusion={v.occlusion}
            />
          )}
        </div>
      </div>
    </div>
  );
}
