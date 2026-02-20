import time
import logging
import cv2
import numpy as np

from models.registry import ModelRegistry
from state.session import SessionState, ActiveSessionState, VerificationStep, ChallengeStep
from processing.face_detection import detect_face
from processing.blink_mouth import update_blink_counts
from processing.occlusion import predict_occlusion
from processing.antispoof import predict_ensemble
from schemas.messages import (
    FrameResponse, VerificationResponse,
    BBox, OcclusionResult, LandmarkResult, AntiSpoofResult, ChallengeResult,
)

logger = logging.getLogger("uvicorn.error")

CAPTURE_FRAMES = 5


def process_frame(frame_bgr: np.ndarray, registry: ModelRegistry, session: SessionState, landmarker) -> dict:
    """Process a single frame through the full pipeline. Returns a JSON-serializable dict."""
    t_start = time.perf_counter()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t0 = time.perf_counter()
    landmarks, bbox = detect_face(landmarker, frame_rgb)
    t1 = time.perf_counter()
    logger.info(f"[Pipeline] face_detect: {(t1-t0)*1000:.0f}ms, step={session.step.value}")

    # No face detected
    if landmarks is None:
        if session.step not in (VerificationStep.RESULT,):
            session.step = VerificationStep.WAITING_FOR_FACE
            session.consecutive_clear_frames = 0
            session.capture_countdown = 0
        return FrameResponse(
            step=session.step.value,
            face_detected=False,
        ).model_dump()

    # Update blink / mouth
    update_blink_counts(landmarks, session)

    landmark_result = LandmarkResult(
        left_blink_count=session.left_blink_count,
        right_blink_count=session.right_blink_count,
        mouth_open=session.mouth_open,
    )

    bbox_model = BBox(**bbox)

    # --- State machine ---

    if session.step == VerificationStep.WAITING_FOR_FACE:
        session.step = VerificationStep.CHECKING_OCCLUSION
        session.consecutive_clear_frames = 0

    if session.step == VerificationStep.CHECKING_OCCLUSION:
        # Run occlusion check
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        face_crop = frame_bgr[y:y+h, x:x+w]
        # Debug: save face crop and full frame
        cv2.imwrite("face_crop.png", face_crop)
        cv2.imwrite("full_frame.png", frame_bgr)
        t2 = time.perf_counter()
        if face_crop.size > 0:
            occ_pred, occ_conf = predict_occlusion(
                face_crop, registry.occlusion_model,
                registry.occlusion_transform, registry.device,
            )
        else:
            occ_pred, occ_conf = 0, 0.0
        t3 = time.perf_counter()

        occ_result = OcclusionResult(is_occluded=bool(occ_pred == 1), confidence=occ_conf)
        logger.info(f"[Pipeline] occlusion: {(t3-t2)*1000:.0f}ms, pred={occ_pred}, conf={occ_conf:.2f}, clear_streak={session.consecutive_clear_frames}, total_frame: {(t3-t_start)*1000:.0f}ms")

        if occ_pred == 0:
            session.consecutive_clear_frames += 1
        else:
            session.consecutive_clear_frames = 0

        if session.consecutive_clear_frames >= session.clear_frames_required:
            session.step = VerificationStep.CAPTURING
            session.capture_countdown = CAPTURE_FRAMES

        return FrameResponse(
            step=session.step.value,
            face_detected=True,
            bbox=bbox_model,
            occlusion=occ_result,
            landmarks=landmark_result,
        ).model_dump()

    if session.step == VerificationStep.CAPTURING:
        session.capture_countdown -= 1
        if session.capture_countdown <= 0:
            session.step = VerificationStep.VERIFYING
            # Run anti-spoof on this frame
            t4 = time.perf_counter()
            label, confidence = predict_ensemble(frame_bgr, bbox, registry)
            t5 = time.perf_counter()
            logger.info(f"[Pipeline] antispoof: {(t5-t4)*1000:.0f}ms, label={label}, conf={confidence:.3f}")
            session.spoof_label = label
            session.spoof_confidence = confidence
            session.step = VerificationStep.RESULT

            is_real = label == 1
            return VerificationResponse(
                passed=is_real,
                is_real=is_real,
                confidence=confidence,
            ).model_dump()

        return FrameResponse(
            step=session.step.value,
            face_detected=True,
            bbox=bbox_model,
            landmarks=landmark_result,
        ).model_dump()

    if session.step == VerificationStep.RESULT:
        is_real = session.spoof_label == 1
        return VerificationResponse(
            passed=is_real,
            is_real=is_real,
            confidence=session.spoof_confidence or 0.0,
        ).model_dump()

    # Fallback
    return FrameResponse(
        step=session.step.value,
        face_detected=True,
        bbox=bbox_model,
        landmarks=landmark_result,
    ).model_dump()


MOUTH_HOLD_REQUIRED = 2.0


def process_frame_active(frame_bgr: np.ndarray, registry: ModelRegistry, session: ActiveSessionState, landmarker) -> dict:
    """Process a single frame through the active verification pipeline with liveness challenge."""
    t_start = time.perf_counter()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    landmarks, bbox = detect_face(landmarker, frame_rgb)

    # No face detected
    if landmarks is None:
        if session.step not in (VerificationStep.RESULT,):
            session.step = VerificationStep.WAITING_FOR_FACE
            session.consecutive_clear_frames = 0
            session.capture_countdown = 0
        return FrameResponse(
            step=session.step.value,
            face_detected=False,
        ).model_dump()

    # Update blink / mouth
    update_blink_counts(landmarks, session)

    # Majority vote over last 5 frames to smooth mouth_open
    session.mouth_history.append(session.mouth_open)
    mouth_open_smoothed = sum(session.mouth_history) > len(session.mouth_history) // 2

    landmark_result = LandmarkResult(
        left_blink_count=session.left_blink_count,
        right_blink_count=session.right_blink_count,
        mouth_open=mouth_open_smoothed,
    )
    bbox_model = BBox(**bbox)

    # --- State machine ---

    if session.step == VerificationStep.WAITING_FOR_FACE:
        session.step = VerificationStep.CHECKING_OCCLUSION
        session.consecutive_clear_frames = 0

    if session.step == VerificationStep.CHECKING_OCCLUSION:
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        face_crop = frame_bgr[y:y+h, x:x+w]
        cv2.imwrite("active_face_crop.png", face_crop)
        cv2.imwrite("active_full_frame.png", frame_bgr)
        if face_crop.size > 0:
            occ_pred, occ_conf = predict_occlusion(
                face_crop, registry.occlusion_model,
                registry.occlusion_transform, registry.device,
            )
        else:
            occ_pred, occ_conf = 0, 0.0

        occ_result = OcclusionResult(is_occluded=bool(occ_pred == 1), confidence=occ_conf)

        if occ_pred == 0:
            session.consecutive_clear_frames += 1
        else:
            session.consecutive_clear_frames = 0

        if session.consecutive_clear_frames >= session.clear_frames_required:
            session.step = VerificationStep.CHALLENGE
            session.challenge_step = ChallengeStep.IN_PROGRESS

        return FrameResponse(
            step=session.step.value,
            face_detected=True,
            bbox=bbox_model,
            occlusion=occ_result,
            landmarks=landmark_result,
        ).model_dump()

    if session.step == VerificationStep.CHALLENGE:
        cv2.imwrite("active_challenge_frame.png", frame_bgr)
        mouth_hold_duration = 0.0

        logger.info(f"[Active CHALLENGE] raw_mouth={session.mouth_open}, history={list(session.mouth_history)}, smoothed={mouth_open_smoothed}, timer_start={session.mouth_open_start}")

        if mouth_open_smoothed:
            if session.mouth_open_start is None:
                session.mouth_open_start = time.time()
                logger.info(f"[Active CHALLENGE] Timer STARTED at {session.mouth_open_start}")
            mouth_hold_duration = time.time() - session.mouth_open_start
            logger.info(f"[Active CHALLENGE] Timer running: {mouth_hold_duration:.2f}s / {MOUTH_HOLD_REQUIRED}s")
            if mouth_hold_duration >= MOUTH_HOLD_REQUIRED:
                logger.info(f"[Active] Challenge complete: mouth held open {mouth_hold_duration:.1f}s")
                session.challenge_step = ChallengeStep.CHALLENGE_DONE
                session.step = VerificationStep.CAPTURING
                session.capture_countdown = CAPTURE_FRAMES
        else:
            if session.mouth_open_start is not None:
                logger.info(f"[Active CHALLENGE] Timer RESET (smoothed=False)")
            session.mouth_open_start = None

        challenge_result = ChallengeResult(
            challenge_step=session.challenge_step.value,
            mouth_open=mouth_open_smoothed,
            mouth_hold_duration=mouth_hold_duration,
        )

        return FrameResponse(
            step=session.step.value,
            face_detected=True,
            bbox=bbox_model,
            landmarks=landmark_result,
            challenge=challenge_result,
        ).model_dump()

    if session.step == VerificationStep.CAPTURING:
        session.capture_countdown -= 1
        if session.capture_countdown <= 0:
            session.step = VerificationStep.VERIFYING
            t4 = time.perf_counter()
            label, confidence = predict_ensemble(frame_bgr, bbox, registry)
            t5 = time.perf_counter()
            logger.info(f"[Active] antispoof: {(t5-t4)*1000:.0f}ms, label={label}, conf={confidence:.3f}")
            session.spoof_label = label
            session.spoof_confidence = confidence
            session.step = VerificationStep.RESULT

            is_real = label == 1
            return VerificationResponse(
                passed=is_real,
                is_real=is_real,
                confidence=confidence,
            ).model_dump()

        return FrameResponse(
            step=session.step.value,
            face_detected=True,
            bbox=bbox_model,
            landmarks=landmark_result,
        ).model_dump()

    if session.step == VerificationStep.RESULT:
        is_real = session.spoof_label == 1
        return VerificationResponse(
            passed=is_real,
            is_real=is_real,
            confidence=session.spoof_confidence or 0.0,
        ).model_dump()

    # Fallback
    return FrameResponse(
        step=session.step.value,
        face_detected=True,
        bbox=bbox_model,
        landmarks=landmark_result,
    ).model_dump()
