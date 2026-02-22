import logging
import cv2
import numpy as np

from models.registry import ModelRegistry
from state.flash_session import FlashSessionState, FlashStep
from processing.face_detection import detect_face
from processing.blink_mouth import update_blink_counts
from processing.occlusion import predict_occlusion
from schemas.messages import FrameResponse, BBox, OcclusionResult, LandmarkResult

logger = logging.getLogger("uvicorn.error")


MIN_FACE_RATIO = 0.28  # face must be at least 28% of frame width


def process_frame_flash(
    frame_bgr: np.ndarray,
    registry: ModelRegistry,
    session: FlashSessionState,
    landmarker,
) -> dict:
    """Phase 1: face detection + occlusion + size check. Transitions to READY_FOR_FLASH."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    landmarks, bbox = detect_face(landmarker, frame_rgb)

    if landmarks is None:
        if session.step != FlashStep.READY_FOR_FLASH:
            session.step = FlashStep.WAITING_FOR_FACE
            session.consecutive_clear_frames = 0
        return FrameResponse(
            step=session.step.value,
            face_detected=False,
        ).model_dump()

    update_blink_counts(landmarks, session)

    landmark_result = LandmarkResult(
        left_blink_count=session.left_blink_count,
        right_blink_count=session.right_blink_count,
        mouth_open=session.mouth_open,
    )
    bbox_model = BBox(**bbox)

    # Check if face is large enough in frame
    img_w = frame_bgr.shape[1]
    face_ratio = bbox["width"] / img_w
    too_small = face_ratio < MIN_FACE_RATIO

    if session.step == FlashStep.WAITING_FOR_FACE:
        session.step = FlashStep.CHECKING_OCCLUSION
        session.consecutive_clear_frames = 0

    if session.step == FlashStep.CHECKING_OCCLUSION:
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        face_crop = frame_bgr[y:y+h, x:x+w]
        if face_crop.size > 0:
            occ_pred, occ_conf = predict_occlusion(
                face_crop, registry.occlusion_model,
                registry.occlusion_transform, registry.device,
            )
        else:
            occ_pred, occ_conf = 0, 0.0

        occ_result = OcclusionResult(is_occluded=bool(occ_pred == 1), confidence=occ_conf)

        # Only count clear frames if face is not occluded AND large enough
        if occ_pred == 0 and not too_small:
            session.consecutive_clear_frames += 1
        else:
            session.consecutive_clear_frames = 0

        if session.consecutive_clear_frames >= session.clear_frames_required:
            session.step = FlashStep.READY_FOR_FLASH

        return FrameResponse(
            step=session.step.value,
            face_detected=True,
            bbox=bbox_model,
            occlusion=occ_result,
            landmarks=landmark_result,
            face_too_small=too_small,
        ).model_dump()

    # READY_FOR_FLASH: keep returning face tracking data
    return FrameResponse(
        step=session.step.value,
        face_detected=True,
        bbox=bbox_model,
        landmarks=landmark_result,
        face_too_small=too_small,
    ).model_dump()
