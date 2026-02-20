export type VerificationStep =
  | "WAITING_FOR_FACE"
  | "CHECKING_OCCLUSION"
  | "CHALLENGE"
  | "CAPTURING"
  | "VERIFYING"
  | "RESULT";

export interface BBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface OcclusionResult {
  is_occluded: boolean;
  confidence: number;
}

export interface LandmarkResult {
  left_blink_count: number;
  right_blink_count: number;
  mouth_open: boolean;
}

export interface AntiSpoofResult {
  label: number;
  confidence: number;
  is_real: boolean;
}

export interface ChallengeStatus {
  challenge_step: "IN_PROGRESS" | "CHALLENGE_DONE";
  mouth_open: boolean;
  mouth_hold_duration: number;
}

export interface FrameResult {
  type: "frame_result";
  step: VerificationStep;
  face_detected: boolean;
  bbox: BBox | null;
  occlusion: OcclusionResult | null;
  landmarks: LandmarkResult | null;
  antispoof: AntiSpoofResult | null;
  challenge: ChallengeStatus | null;
}

export interface VerificationResult {
  type: "verification_result";
  step: "RESULT";
  passed: boolean;
  is_real: boolean;
  confidence: number;
}

export interface ResetAck {
  type: "reset_ack";
  step: VerificationStep;
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

export type ServerMessage = FrameResult | VerificationResult | ResetAck | ErrorMessage;
