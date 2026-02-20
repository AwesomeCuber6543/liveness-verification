"use client";
import { useRef, useState, useCallback, useEffect } from "react";
import { FRAME_WIDTH, FRAME_HEIGHT, JPEG_QUALITY } from "@/lib/constants";

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mountedRef = useRef(false);
  // Monotonic counter — each startCamera call gets an ID. If a newer call
  // has been issued by the time the async work finishes, the older one bails.
  const startIdRef = useRef(0);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const startCamera = useCallback(async () => {
    const myId = ++startIdRef.current;
    console.log("[Camera] startCamera called, id:", myId);

    // Already running
    if (streamRef.current && streamRef.current.active) {
      console.log("[Camera] Stream already active, reusing");
      if (videoRef.current && !videoRef.current.srcObject) {
        videoRef.current.srcObject = streamRef.current;
        await videoRef.current.play();
      }
      setIsReady(true);
      return;
    }

    // Clean up dead stream
    if (streamRef.current && !streamRef.current.active) {
      console.log("[Camera] Dead stream found, cleaning up");
      streamRef.current = null;
    }

    setError(null);
    try {
      console.log("[Camera] Requesting getUserMedia...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: FRAME_WIDTH, height: FRAME_HEIGHT, facingMode: "user" },
      });

      // Check if this call is still the latest one and component is still mounted
      if (startIdRef.current !== myId || !mountedRef.current) {
        console.log("[Camera] Stale startCamera (id:", myId, "current:", startIdRef.current, "mounted:", mountedRef.current, "), stopping acquired stream");
        stream.getTracks().forEach((t) => t.stop());
        return;
      }

      console.log("[Camera] getUserMedia success, tracks:", stream.getTracks().map(t => ({ kind: t.kind, state: t.readyState })));

      // Stop any stream that snuck in while we were waiting
      if (streamRef.current && streamRef.current !== stream) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        console.log("[Camera] Video playing");
        setIsReady(true);
      } else {
        console.warn("[Camera] videoRef.current is null, stopping stream");
        stream.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
    } catch (err) {
      // If this is a stale call, don't set error state
      if (startIdRef.current !== myId || !mountedRef.current) {
        console.log("[Camera] Ignoring error from stale startCamera call");
        return;
      }

      console.error("[Camera] getUserMedia error:", err);
      const name = (err as DOMException)?.name;
      const message = (err as Error)?.message;
      if (name === "NotAllowedError") {
        setError("Camera access denied. Please allow camera permissions.");
      } else if (name === "NotFoundError") {
        setError("No camera found on this device.");
      } else if (name === "AbortError") {
        // AbortError from play() interruption — retry silently
        console.log("[Camera] AbortError (play interrupted), not setting error");
      } else if (name === "NotReadableError") {
        setError("Camera is in use by another application.");
      } else {
        setError(`Could not access camera: ${name || "unknown"} - ${message || "unknown"}`);
      }
    }
  }, []);

  const stopCamera = useCallback(() => {
    console.log("[Camera] stopCamera called");
    // Bump the ID so any in-flight startCamera knows to bail
    startIdRef.current++;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsReady(false);
  }, []);

  const captureFrame = useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || !isReady) {
        resolve(null);
        return;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        resolve(null);
        return;
      }
      canvas.width = FRAME_WIDTH;
      canvas.height = FRAME_HEIGHT;
      ctx.drawImage(video, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
      canvas.toBlob(
        (blob) => resolve(blob),
        "image/jpeg",
        JPEG_QUALITY
      );
    });
  }, [isReady]);

  useEffect(() => {
    return () => stopCamera();
  }, [stopCamera]);

  return { videoRef, canvasRef, isReady, error, startCamera, stopCamera, captureFrame };
}
