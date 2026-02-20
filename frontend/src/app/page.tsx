export default function Home() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 dark:bg-zinc-950">
      <main className="flex flex-col items-center gap-12 px-6 py-16">
        <div className="text-center">
          <h1 className="text-4xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50">
            Face Verification
          </h1>
          <p className="mt-3 text-lg text-zinc-500 dark:text-zinc-400">
            Choose a verification method to get started
          </p>
        </div>

        <div className="flex flex-col gap-6 sm:flex-row">
          <a
            href="/verify/passive"
            className="flex h-48 w-72 flex-col items-center justify-center gap-4 rounded-2xl border border-zinc-200 bg-white p-8 shadow-sm transition-all hover:border-zinc-300 hover:shadow-md dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-zinc-700"
          >
            <div className="flex h-14 w-14 items-center justify-center rounded-full bg-emerald-100 dark:bg-emerald-900/30">
              <svg
                className="h-7 w-7 text-emerald-600 dark:text-emerald-400"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 12.75 11.25 15 15 9.75m-3-7.036A11.959 11.959 0 0 1 3.598 6 11.99 11.99 0 0 0 3 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285Z"
                />
              </svg>
            </div>
            <div className="text-center">
              <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">
                Passive Verification
              </h2>
              <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
                Quick, automatic check
              </p>
            </div>
          </a>

          <a
            href="/verify/active"
            className="flex h-48 w-72 flex-col items-center justify-center gap-4 rounded-2xl border border-zinc-200 bg-white p-8 shadow-sm transition-all hover:border-zinc-300 hover:shadow-md dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-zinc-700"
          >
            <div className="flex h-14 w-14 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900/30">
              <svg
                className="h-7 w-7 text-blue-600 dark:text-blue-400"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M15.75 5.25a3 3 0 0 1 3 3m3 0a6 6 0 0 1-7.029 5.912c-.563-.097-1.159.026-1.563.43L10.5 17.25H8.25v2.25H6v2.25H2.25v-2.818c0-.597.237-1.17.659-1.591l6.499-6.499c.404-.404.527-1 .43-1.563A6 6 0 1 1 21.75 8.25Z"
                />
              </svg>
            </div>
            <div className="text-center">
              <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">
                Active Verification
              </h2>
              <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
                Challenge-based check
              </p>
            </div>
          </a>
        </div>
      </main>
    </div>
  );
}
