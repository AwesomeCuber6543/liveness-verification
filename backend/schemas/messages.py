from pydantic import BaseModel


class BBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class OcclusionResult(BaseModel):
    is_occluded: bool
    confidence: float


class LandmarkResult(BaseModel):
    left_blink_count: int
    right_blink_count: int
    mouth_open: bool


class AntiSpoofResult(BaseModel):
    label: int
    confidence: float
    is_real: bool


class ChallengeResult(BaseModel):
    challenge_step: str
    mouth_open: bool
    mouth_hold_duration: float


class FrameResponse(BaseModel):
    type: str = "frame_result"
    step: str
    face_detected: bool
    bbox: BBox | None = None
    occlusion: OcclusionResult | None = None
    landmarks: LandmarkResult | None = None
    antispoof: AntiSpoofResult | None = None
    challenge: ChallengeResult | None = None


class VerificationResponse(BaseModel):
    type: str = "verification_result"
    step: str = "RESULT"
    passed: bool
    is_real: bool
    confidence: float
