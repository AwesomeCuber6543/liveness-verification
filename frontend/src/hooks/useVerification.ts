"use client";
import { useState, useEffect, useCallback, useRef } from "react";
import { useCamera } from "./useCamera";
import { useWebSocket } from "./useWebSocket";
import { WS_URL, WS_URL_ACTIVE } from "@/lib/constants";
import type {
  VerificationStep,
  BBox,
  OcclusionResult,
  LandmarkResult,
  ChallengeStatus,
  VerificationResult,
} from "@/lib/types";

export function useVerification(mode: "passive" | "active" = "passive") {
  const wsUrl = mode === "active" ? WS_URL_ACTIVE : WS_URL;
  const camera = useCamera();
  const ws = useWebSocket(wsUrl);

  const [step, setStep] = useState<VerificationStep>("WAITING_FOR_FACE");
  const [bbox, setBbox] = useState<BBox | null>(null);
  const [occlusion, setOcclusion] = useState<OcclusionResult | null>(null);
  const [landmarks, setLandmarks] = useState<LandmarkResult | null>(null);
  const [challenge, setChallenge] = useState<ChallengeStatus | null>(null);
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [faceDetected, setFaceDetected] = useState(false);

  const pendingRef = useRef(false);
  const activeRef = useRef(false);
  const stepRef = useRef<VerificationStep>("WAITING_FOR_FACE");

  // Process server messages
  useEffect(() => {
    if (!ws.lastMessage) return;
    const msg = ws.lastMessage;

    console.log("[Verify] Server message:", msg.type, "step" in msg ? (msg as { step: string }).step : "");

    if (msg.type === "frame_result") {
      console.log("[Verify] frame_result step:", msg.step, "face:", msg.face_detected, "challenge:", JSON.stringify(msg.challenge));
      setStep(msg.step);
      stepRef.current = msg.step;
      setFaceDetected(msg.face_detected);
      setBbox(msg.bbox ?? null);
      setOcclusion(msg.occlusion ?? null);
      setLandmarks(msg.landmarks ?? null);
      setChallenge(msg.challenge ?? null);
    } else if (msg.type === "verification_result") {
      setStep("RESULT");
      stepRef.current = "RESULT";
      setResult(msg as VerificationResult);
    } else if (msg.type === "reset_ack") {
      setStep(msg.step);
      stepRef.current = msg.step;
      setResult(null);
      setBbox(null);
      setOcclusion(null);
      setChallenge(null);
    }

    pendingRef.current = false;
  }, [ws.lastMessage]);

  // Frame capture loop with backpressure
  useEffect(() => {
    if (!camera.isReady || !ws.isConnected) {
      console.log("[Verify] Loop not starting: cameraReady:", camera.isReady, "wsConnected:", ws.isConnected);
      return;
    }
    console.log("[Verify] Starting frame capture loop");
    activeRef.current = true;

    const loop = async () => {
      while (activeRef.current) {
        if (!pendingRef.current && stepRef.current !== "RESULT") {
          const blob = await camera.captureFrame();
          if (blob && activeRef.current) {
            pendingRef.current = true;
            ws.sendFrame(blob);
          }
        }
        await new Promise((r) => setTimeout(r, 30));
      }
      console.log("[Verify] Frame capture loop stopped");
    };
    loop();

    return () => {
      console.log("[Verify] Cleaning up frame capture loop");
      activeRef.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera.isReady, ws.isConnected]);

  const start = useCallback(async () => {
    console.log("[Verify] start() called");
    await camera.startCamera();
    ws.connect();
  }, [camera, ws]);

  const reset = useCallback(() => {
    console.log("[Verify] reset() called");
    ws.sendCommand({ type: "reset" });
    setResult(null);
    setStep("WAITING_FOR_FACE");
    setBbox(null);
    setOcclusion(null);
    setChallenge(null);
    pendingRef.current = false;
  }, [ws]);

  const stop = useCallback(() => {
    console.log("[Verify] stop() called");
    activeRef.current = false;
    camera.stopCamera();
    ws.disconnect();
  }, [camera, ws]);

  return {
    // Camera refs
    videoRef: camera.videoRef,
    canvasRef: camera.canvasRef,
    cameraReady: camera.isReady,
    cameraError: camera.error,
    // WS state
    connected: ws.isConnected,
    wsError: ws.error,
    // Verification state
    step,
    faceDetected,
    bbox,
    occlusion,
    landmarks,
    challenge,
    result,
    // Actions
    start,
    reset,
    stop,
  };
}
