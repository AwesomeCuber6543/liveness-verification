"use client";
import { useEffect, useRef, useState } from "react";
import type { ChallengeStatus } from "@/lib/types";

interface ChallengeChecklistProps {
  challenge: ChallengeStatus | null;
}

export function ChallengeChecklist({ challenge }: ChallengeChecklistProps) {
  const done = challenge?.challenge_step === "CHALLENGE_DONE";
  const mouthOpen = challenge?.mouth_open ?? false;
  const serverDuration = challenge?.mouth_hold_duration ?? 0;

  // Interpolate progress client-side for smooth bar
  const [progress, setProgress] = useState(0);
  const lastServerTime = useRef(0);
  const lastServerDuration = useRef(0);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    if (done) {
      setProgress(1);
      return;
    }

    if (!mouthOpen || serverDuration === 0) {
      setProgress(0);
      lastServerDuration.current = 0;
      return;
    }

    // Got a new server update with duration > 0
    lastServerTime.current = performance.now();
    lastServerDuration.current = serverDuration;

    const tick = () => {
      const elapsed = (performance.now() - lastServerTime.current) / 1000;
      const estimated = lastServerDuration.current + elapsed;
      setProgress(Math.min(estimated / 2.0, 1));
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(rafRef.current);
  }, [serverDuration, mouthOpen, done]);

  return (
    <div className="w-full rounded-2xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-zinc-100 dark:bg-zinc-800">
          {done ? (
            <svg className="h-5 w-5 text-emerald-500" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
            </svg>
          ) : (
            <span className="text-lg">👄</span>
          )}
        </div>
        <div className="flex-1">
          <p className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
            Open your mouth wide
          </p>
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            {done ? "Complete" : mouthOpen ? `Hold... ${(progress * 2).toFixed(1)}s / 2.0s` : "Open wide and hold for 2 seconds"}
          </p>
        </div>
      </div>

      <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
        <div
          className="h-full rounded-full bg-blue-500"
          style={{ width: `${progress * 100}%` }}
        />
      </div>
    </div>
  );
}
