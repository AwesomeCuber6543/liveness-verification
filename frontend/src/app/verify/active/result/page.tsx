"use client";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";

function ResultContent() {
  const searchParams = useSearchParams();
  const passed = searchParams.get("passed") === "true";

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-zinc-50 px-6 dark:bg-zinc-950">
      <div className="flex w-full max-w-sm flex-col items-center gap-8">
        <div
          className={`flex h-20 w-20 items-center justify-center rounded-full ${
            passed
              ? "bg-emerald-100 dark:bg-emerald-900/30"
              : "bg-red-100 dark:bg-red-900/30"
          }`}
        >
          {passed ? (
            <svg
              className="h-10 w-10 text-emerald-600 dark:text-emerald-400"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={2}
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
            </svg>
          ) : (
            <svg
              className="h-10 w-10 text-red-600 dark:text-red-400"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={2}
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
            </svg>
          )}
        </div>

        <div className="text-center">
          <h1
            className={`text-2xl font-bold ${
              passed
                ? "text-emerald-700 dark:text-emerald-400"
                : "text-red-700 dark:text-red-400"
            }`}
          >
            {passed ? "Verification Passed" : "Verification Failed"}
          </h1>
          <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
            {passed
              ? "Your identity has been verified successfully."
              : "We couldn't verify your identity. Please try again."}
          </p>
        </div>

        <div className="flex flex-col gap-3 w-full">
          {!passed && (
            <a
              href="/verify/active"
              className="flex h-11 items-center justify-center rounded-full bg-zinc-900 text-sm font-medium text-white transition-colors hover:bg-zinc-800 dark:bg-zinc-50 dark:text-zinc-900 dark:hover:bg-zinc-200"
            >
              Try Again
            </a>
          )}
          <a
            href="/"
            className="flex h-11 items-center justify-center rounded-full border border-zinc-200 text-sm font-medium text-zinc-700 transition-colors hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
          >
            Back to Home
          </a>
        </div>
      </div>
    </div>
  );
}

export default function ResultPage() {
  return (
    <Suspense>
      <ResultContent />
    </Suspense>
  );
}
