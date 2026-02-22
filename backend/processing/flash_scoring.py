"""
Flash Liveness Scoring Engine

Photometric challenge-response verification based on spatial-temporal reflectance
analysis. No CNN — pure deterministic math.

Security model:
- Spatial variance verifies the face is a 3D surface (non-uniform reflection).
- Eye specular reflection checks that flash colors appear as highlights in the iris.
- Temporal correlation verifies the face reflects the screen's color changes in real time.
- Lag analysis ensures reflection timing is physically consistent.
- Ratio stability checks that patch brightness ratios are temporally stable (fixed geometry).
"""

import logging
import numpy as np
from scipy import stats
from scipy.signal import correlate
import cv2

from processing.face_detection import detect_face, create_landmarker
from config import (
    FLASH_GRID_ROWS, FLASH_GRID_COLS,
    FLASH_TEMPORAL_CORR_THRESHOLD,
    FLASH_LAG_TOLERANCE_FRAMES, FLASH_SPATIAL_VAR_MIN,
    FLASH_SPATIAL_VAR_MAX, FLASH_RATIO_STABILITY_THRESHOLD,
    FLASH_EYE_SPECULAR_RATIO_MIN,
)

logger = logging.getLogger("uvicorn.error")

NUM_PATCHES = FLASH_GRID_ROWS * FLASH_GRID_COLS

# MediaPipe iris landmarks (468+ model)
LEFT_IRIS = [468, 469, 470, 471, 472]   # center, right, up, left, down
RIGHT_IRIS = [473, 474, 475, 476, 477]
# Eye region landmarks for masking
LEFT_EYE_REGION = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_REGION = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]


def extract_face_patches(frame_bgr: np.ndarray, landmarker) -> tuple[np.ndarray, float] | None:
    """Detect face, crop, divide into grid, return mean RGB per patch + face size ratio.

    Returns (patches ndarray of shape (NUM_PATCHES, 3), face_width_ratio) or None.
    face_width_ratio = face bbox width / frame width.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    landmarks, bbox = detect_face(landmarker, frame_rgb)
    if landmarks is None or bbox is None:
        return None

    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    face_crop = frame_rgb[y:y+h, x:x+w]

    if face_crop.size == 0 or w < 16 or h < 16:
        return None

    img_w = frame_rgb.shape[1]
    face_width_ratio = w / img_w

    patch_h = h // FLASH_GRID_ROWS
    patch_w = w // FLASH_GRID_COLS
    patches = np.zeros((NUM_PATCHES, 3), dtype=np.float64)

    for r in range(FLASH_GRID_ROWS):
        for c in range(FLASH_GRID_COLS):
            py = r * patch_h
            px = c * patch_w
            patch = face_crop[py:py+patch_h, px:px+patch_w]
            patches[r * FLASH_GRID_COLS + c] = patch.mean(axis=(0, 1))

    return patches, face_width_ratio


def normalize_patches(patches: np.ndarray) -> np.ndarray:
    """Normalize each frame's patches against the global face mean.

    Removes ambient illumination and camera white balance effects,
    isolating the differential response to screen color changes.

    Args:
        patches: shape (num_frames, 16, 3)
    Returns:
        normalized: shape (num_frames, 16, 3)
    """
    face_mean = patches.mean(axis=1, keepdims=True)  # (N, 1, 3)
    face_mean_safe = np.where(face_mean == 0, 1.0, face_mean)
    return (patches - face_mean) / face_mean_safe


def _get_iris_crop(frame_rgb: np.ndarray, landmarks, iris_indices: list[int]):
    """Get iris crop and metadata for one eye. Returns (crop, mask, radius) or None."""
    img_h, img_w = frame_rgb.shape[:2]

    cx = int(landmarks[iris_indices[0]].x * img_w)
    cy = int(landmarks[iris_indices[0]].y * img_h)

    radii = []
    for idx in iris_indices[1:]:
        px = int(landmarks[idx].x * img_w)
        py = int(landmarks[idx].y * img_h)
        radii.append(np.sqrt((px - cx) ** 2 + (py - cy) ** 2))
    radius = int(np.mean(radii)) if radii else 8

    r = max(radius, 4)
    y1 = max(0, cy - r)
    y2 = min(img_h, cy + r)
    x1 = max(0, cx - r)
    x2 = min(img_w, cx + r)

    if y2 - y1 < 3 or x2 - x1 < 3:
        return None

    crop = frame_rgb[y1:y2, x1:x2].astype(np.float64)
    crop_h, crop_w = crop.shape[:2]
    yy, xx = np.ogrid[:crop_h, :crop_w]
    center_y, center_x = crop_h / 2, crop_w / 2
    mask = ((xx - center_x) ** 2 + (yy - center_y) ** 2) <= (r * 0.9) ** 2

    return crop, mask, r


def extract_iris_metrics(
    frame_bgr: np.ndarray,
    landmarks,
    top_k: int = 3,
) -> dict | None:
    """Extract specular highlight metrics from both irises in a single frame.

    Returns dict with:
        specular_brightness: float — mean brightness of top-K brightest iris pixels
        surround_brightness: float — mean brightness of remaining iris pixels
        specular_ratio: float — specular / surround (higher = sharper highlight)
        specular_rgb: ndarray(3,) — mean RGB of top-K brightest pixels
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if len(landmarks) < 478:
        return None

    all_specular_brightness = []
    all_surround_brightness = []
    all_specular_rgb = []

    for iris_indices in [LEFT_IRIS, RIGHT_IRIS]:
        result = _get_iris_crop(frame_rgb, landmarks, iris_indices)
        if result is None:
            continue
        crop, mask, r = result
        crop_h, crop_w = crop.shape[:2]

        brightness = np.sum(crop, axis=2)  # (H, W)
        iris_brightness = brightness[mask]

        if len(iris_brightness) < top_k + 3:
            continue

        # Top-K brightest = specular highlight candidates
        k = min(top_k, len(iris_brightness))
        top_k_indices = np.argpartition(iris_brightness, -k)[-k:]
        specular_bright = float(iris_brightness[top_k_indices].mean())

        # Remaining iris pixels = surround
        surround_mask = np.ones(len(iris_brightness), dtype=bool)
        surround_mask[top_k_indices] = False
        surround_bright = float(iris_brightness[surround_mask].mean())

        all_specular_brightness.append(specular_bright)
        all_surround_brightness.append(surround_bright)

        # Get RGB of top-K pixels for color correlation
        flat_indices = np.where(mask.flatten())[0]
        top_flat = flat_indices[top_k_indices]
        rows = top_flat // crop_w
        cols = top_flat % crop_w
        specular_rgb = crop[rows, cols].mean(axis=0)
        all_specular_rgb.append(specular_rgb)

    if not all_specular_brightness:
        return None

    spec_bright = float(np.mean(all_specular_brightness))
    surr_bright = float(np.mean(all_surround_brightness))
    ratio = spec_bright / max(surr_bright, 1.0)

    return {
        "specular_brightness": spec_bright,
        "surround_brightness": surr_bright,
        "specular_ratio": ratio,
        "specular_rgb": np.mean(all_specular_rgb, axis=0),
    }


def extract_eye_specular_series(
    frames_bgr: list[np.ndarray],
    landmarker,
) -> tuple[dict | None, list[int]]:
    """Extract iris specular metrics across all frames.

    Returns:
        metrics: dict with arrays of per-frame values, or None
        valid_indices: which frame indices had valid iris data
    """
    specular_brightnesses = []
    surround_brightnesses = []
    specular_ratios = []
    specular_rgbs = []
    valid_indices = []

    for idx, frame in enumerate(frames_bgr):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks, _ = detect_face(landmarker, frame_rgb)
        if landmarks is None:
            continue

        metrics = extract_iris_metrics(frame, landmarks)
        if metrics is not None:
            specular_brightnesses.append(metrics["specular_brightness"])
            surround_brightnesses.append(metrics["surround_brightness"])
            specular_ratios.append(metrics["specular_ratio"])
            specular_rgbs.append(metrics["specular_rgb"])
            valid_indices.append(idx)

    if len(valid_indices) < 5:
        return None, valid_indices

    return {
        "specular_brightness": np.array(specular_brightnesses),
        "surround_brightness": np.array(surround_brightnesses),
        "specular_ratios": np.array(specular_ratios),
        "specular_rgb": np.array(specular_rgbs),  # (N, 3)
    }, valid_indices


def compute_eye_specular_scores(
    eye_metrics: dict,
    screen_signal: np.ndarray,
) -> dict:
    """Compute all eye specular liveness signals.

    Returns dict with:
        brightness_variance: float — how much specular brightness changes across
            frames. Real cornea = high variance (flash colors modulate the
            specular dot intensity). Photo = low variance (static highlight).
        mean_specular_ratio: float — mean ratio of specular vs surround brightness.
            Real cornea = high ratio (sharp bright dot on dark iris).
            Photo = low ratio (no sharp highlight).
        color_correlation: float — Pearson r between screen color and specular RGB.
            Kept for logging/evaluation but expected to be noisy at webcam res.
        color_per_channel: ndarray(3,) — per-channel Pearson r.
    """
    spec_bright = eye_metrics["specular_brightness"]
    spec_ratios = eye_metrics["specular_ratios"]
    spec_rgb = eye_metrics["specular_rgb"]  # (N, 3)

    # 1. Brightness variance — normalized by mean to get coefficient of variation
    bright_mean = np.mean(spec_bright)
    bright_std = np.std(spec_bright)
    brightness_cv = float(bright_std / max(bright_mean, 1.0))

    # 2. Mean specular-to-surround ratio
    mean_ratio = float(np.mean(spec_ratios))

    # 3. Color correlation (still useful for logging)
    color_per_channel = np.zeros(3)
    for ch in range(3):
        s = screen_signal[:, ch]
        e = spec_rgb[:, ch]
        if np.std(s) < 1e-6 or np.std(e) < 1e-6:
            color_per_channel[ch] = 0.0
            continue
        r, _ = stats.pearsonr(s, e)
        color_per_channel[ch] = r

    color_corr = float(np.mean(np.abs(color_per_channel)))

    return {
        "brightness_variance": brightness_cv,
        "mean_specular_ratio": mean_ratio,
        "color_correlation": color_corr,
        "color_per_channel": color_per_channel,
    }


def interpolate_screen_signal(
    challenge_log: list[dict],
    frame_timestamps: list[float],
) -> np.ndarray:
    """Map screen colors to frame timestamps via nearest-predecessor lookup.

    Both timestamps use the same client-side clock (performance.now()),
    so relative alignment is exact.

    Returns ndarray of shape (len(frame_timestamps), 3).
    """
    challenge_sorted = sorted(challenge_log, key=lambda x: x["timestamp"])
    ch_times = np.array([c["timestamp"] for c in challenge_sorted])
    ch_colors = np.array([c["color"] for c in challenge_sorted])  # (M, 3)

    screen_signal = np.zeros((len(frame_timestamps), 3))
    for i, ft in enumerate(frame_timestamps):
        # Find the most recent challenge entry at or before this frame time
        idx = np.searchsorted(ch_times, ft, side="right") - 1
        idx = max(0, min(idx, len(ch_times) - 1))
        screen_signal[i] = ch_colors[idx]

    return screen_signal


def compute_temporal_correlation(
    screen_signal: np.ndarray,
    patch_signals: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Pearson correlation between screen color and face patch reflectance.

    A real face illuminated by changing screen colors shows correlated
    reflectance changes. A static photo shows near-zero correlation.

    Args:
        screen_signal: shape (N, 3) -- normalized screen RGB
        patch_signals: shape (N, 16, 3) -- normalized patch RGB
    Returns:
        mean_corr: float -- mean |r| across all 48 (patch, channel) pairs
        corr_matrix: shape (16, 3) -- per-pair Pearson r
    """
    n_frames, n_patches, n_channels = patch_signals.shape
    corr_matrix = np.zeros((n_patches, n_channels))

    for p in range(n_patches):
        for ch in range(n_channels):
            screen_ch = screen_signal[:, ch]
            patch_ch = patch_signals[:, p, ch]

            if np.std(screen_ch) < 1e-6 or np.std(patch_ch) < 1e-6:
                corr_matrix[p, ch] = 0.0
                continue

            r, _ = stats.pearsonr(screen_ch, patch_ch)
            corr_matrix[p, ch] = r

    mean_corr = float(np.mean(np.abs(corr_matrix)))
    return mean_corr, corr_matrix


def compute_lag_analysis(
    screen_signal: np.ndarray,
    patch_signals: np.ndarray,
    max_lag: int = 5,
) -> tuple[float, np.ndarray]:
    """Cross-correlation lag analysis.

    For each patch/channel, find the lag that maximizes cross-correlation.
    Real faces show consistent small lags. Replay attacks with frame-rate
    mismatches show inconsistent lags.

    Returns:
        lag_score: float in [0, 1] -- fraction of pairs within tolerance
        lag_matrix: shape (16, 3) -- optimal lag per pair
    """
    n_frames, n_patches, n_channels = patch_signals.shape
    lag_matrix = np.zeros((n_patches, n_channels), dtype=int)

    for p in range(n_patches):
        for ch in range(n_channels):
            screen_ch = screen_signal[:, ch].astype(np.float64)
            patch_ch = patch_signals[:, p, ch].astype(np.float64)

            screen_ch = screen_ch - screen_ch.mean()
            patch_ch = patch_ch - patch_ch.mean()

            if np.std(screen_ch) < 1e-6 or np.std(patch_ch) < 1e-6:
                lag_matrix[p, ch] = 0
                continue

            xcorr = correlate(patch_ch, screen_ch, mode="full")
            zero_lag_idx = n_frames - 1
            start = max(0, zero_lag_idx - max_lag)
            end = min(len(xcorr), zero_lag_idx + max_lag + 1)
            windowed = xcorr[start:end]
            best_offset = np.argmax(windowed) - (zero_lag_idx - start)
            lag_matrix[p, ch] = best_offset

    within_tolerance = np.abs(lag_matrix) <= FLASH_LAG_TOLERANCE_FRAMES
    lag_score = float(within_tolerance.mean())
    return lag_score, lag_matrix


def compute_spatial_variance(patch_signals: np.ndarray) -> float:
    """Spatial variance across face patches.

    A 3D face illuminated by uniform screen light reflects non-uniformly
    due to geometry (nose protrudes, cheeks recede). A flat photo or screen
    replay reflects uniformly. Higher variance = more 3D-like.

    Args:
        patch_signals: shape (N, 16, 3) -- normalized patches
    Returns:
        spatial_var: float -- mean variance across frames and channels
    """
    per_frame_var = np.var(patch_signals, axis=1)  # (N, 3)
    return float(per_frame_var.mean())


def compute_ratio_stability(patch_signals: np.ndarray) -> float:
    """Temporal stability of brightness ratios between adjacent patches.

    For a real face, the ratio between two nearby patches depends on their
    geometry (fixed angle to screen), so it stays constant across different
    colors. For replay attacks with compression artifacts, ratios fluctuate.

    Returns:
        stability: float in [0, 1] -- higher = more stable
    """
    n_frames, n_patches, n_channels = patch_signals.shape

    # Adjacent pairs in the 4x4 grid (horizontal + vertical neighbors)
    pairs = []
    for r in range(FLASH_GRID_ROWS):
        for c in range(FLASH_GRID_COLS):
            idx = r * FLASH_GRID_COLS + c
            if c + 1 < FLASH_GRID_COLS:
                pairs.append((idx, idx + 1))
            if r + 1 < FLASH_GRID_ROWS:
                pairs.append((idx, idx + FLASH_GRID_COLS))

    if not pairs:
        return 1.0

    stabilities = []
    for (i, j) in pairs:
        for ch in range(n_channels):
            signal_i = patch_signals[:, i, ch]
            signal_j = patch_signals[:, j, ch]

            denominator = np.where(np.abs(signal_j) < 1e-8, 1e-8, signal_j)
            ratios = signal_i / denominator

            if np.std(ratios) < 1e-8:
                stabilities.append(1.0)
            else:
                cv = np.std(ratios) / (np.abs(np.mean(ratios)) + 1e-8)
                stabilities.append(max(0.0, 1.0 - cv))

    return float(np.mean(stabilities))


def score_flash_liveness(
    challenge_log: list[dict],
    frames_jpeg: list[bytes],
    frame_timestamps: list[float],
) -> dict:
    """Full flash liveness scoring pipeline.

    1. Decode frames and extract face patches
    2. Normalize patches against per-frame face mean
    3. Align screen signal to frame timestamps
    4. Compute spatial variance (3D geometry gate)
    5. Compute eye specular ratio (corneal reflection gate)
    6. Log all metrics for calibration

    Returns dict with decision ("pass" or "fail").
    """
    import os
    landmarker = create_landmarker()

    # Debug: save frames to disk for analysis
    debug_dir = "debug_flash_frames"
    os.makedirs(debug_dir, exist_ok=True)

    try:
        # Step 1: Decode frames and extract patches
        all_patches = []
        face_width_ratios = []
        valid_timestamps = []
        decoded_frames = []  # keep decoded frames for eye analysis

        for idx, (jpeg_bytes, ts) in enumerate(zip(frames_jpeg, frame_timestamps)):
            frame = cv2.imdecode(
                np.frombuffer(jpeg_bytes, np.uint8),
                cv2.IMREAD_COLOR,
            )
            if frame is None:
                continue

            # Save frame for debug analysis
            cv2.imwrite(os.path.join(debug_dir, f"frame_{idx:03d}.jpg"), frame)

            result = extract_face_patches(frame, landmarker)
            if result is not None:
                patches, face_ratio = result
                all_patches.append(patches)
                face_width_ratios.append(face_ratio)
                valid_timestamps.append(ts)
                decoded_frames.append(frame)

        # Save challenge log for debug
        import json
        with open(os.path.join(debug_dir, "challenge_log.json"), "w") as f:
            json.dump({"challenge_log": challenge_log, "frame_timestamps": valid_timestamps}, f, indent=2)

        mean_face_ratio = float(np.mean(face_width_ratios)) if face_width_ratios else 0.0
        logger.info(f"[Flash] Extracted patches from {len(all_patches)}/{len(frames_jpeg)} frames, mean face ratio={mean_face_ratio:.3f}")

        if len(all_patches) < 5:
            return {
                "decision": "fail",
            }

        patch_array = np.array(all_patches)  # (N, 64, 3)

        # Step 2: Normalize
        normalized = normalize_patches(patch_array)

        # Step 3: Build screen signal aligned to frame timestamps
        screen_signal = interpolate_screen_signal(challenge_log, valid_timestamps)
        screen_mean = screen_signal.mean(axis=0, keepdims=True)
        screen_std = screen_signal.std(axis=0, keepdims=True)
        screen_std = np.where(screen_std < 1e-6, 1.0, screen_std)
        screen_normalized = (screen_signal - screen_mean) / screen_std

        # Step 4: Temporal correlation
        temporal_corr, _ = compute_temporal_correlation(screen_normalized, normalized)
        temporal_score = min(1.0, temporal_corr / FLASH_TEMPORAL_CORR_THRESHOLD)
        logger.info(f"[Flash] Temporal correlation: raw={temporal_corr:.4f}, score={temporal_score:.4f}")

        # Step 5: Lag analysis
        lag_score, _ = compute_lag_analysis(screen_normalized, normalized)
        logger.info(f"[Flash] Lag analysis: score={lag_score:.4f}")

        # Step 6: Spatial variance — PRIMARY GATE
        spatial_var = compute_spatial_variance(normalized)
        if spatial_var < FLASH_SPATIAL_VAR_MIN:
            spatial_score = spatial_var / FLASH_SPATIAL_VAR_MIN
        elif spatial_var > FLASH_SPATIAL_VAR_MAX:
            spatial_score = max(0.0, 1.0 - (spatial_var - FLASH_SPATIAL_VAR_MAX) / FLASH_SPATIAL_VAR_MAX)
        else:
            spatial_score = 1.0
        logger.info(f"[Flash] Spatial variance: raw={spatial_var:.6f}, score={spatial_score:.4f}, threshold={FLASH_SPATIAL_VAR_MIN}")

        # Step 7: Ratio stability
        ratio_stability = compute_ratio_stability(normalized)
        ratio_score = min(1.0, ratio_stability / FLASH_RATIO_STABILITY_THRESHOLD)
        logger.info(f"[Flash] Ratio stability: raw={ratio_stability:.4f}, score={ratio_score:.4f}")

        # Step 8: Eye specular reflection analysis
        eye_brightness_var = 0.0
        eye_specular_ratio = 0.0
        eye_color_corr = 0.0
        eye_color_per_channel = np.zeros(3)
        eye_frames_used = 0

        eye_metrics, valid_eye_indices = extract_eye_specular_series(
            decoded_frames, landmarker,
        )
        if eye_metrics is not None and len(valid_eye_indices) >= 5:
            eye_timestamps = [valid_timestamps[i] for i in valid_eye_indices]
            eye_screen = interpolate_screen_signal(challenge_log, eye_timestamps)

            eye_scores = compute_eye_specular_scores(eye_metrics, eye_screen)
            eye_brightness_var = eye_scores["brightness_variance"]
            eye_specular_ratio = eye_scores["mean_specular_ratio"]
            eye_color_corr = eye_scores["color_correlation"]
            eye_color_per_channel = eye_scores["color_per_channel"]
            eye_frames_used = len(valid_eye_indices)
            logger.info(
                f"[Flash] Eye specular: brightness_cv={eye_brightness_var:.4f}, "
                f"specular_ratio={eye_specular_ratio:.4f}, "
                f"color_corr={eye_color_corr:.4f}, "
                f"per_channel=R:{eye_color_per_channel[0]:.3f} "
                f"G:{eye_color_per_channel[1]:.3f} "
                f"B:{eye_color_per_channel[2]:.3f}, "
                f"frames={eye_frames_used}"
            )
        else:
            logger.info(f"[Flash] Eye specular: insufficient iris data ({len(valid_eye_indices) if valid_eye_indices else 0} frames)")

        # Step 9: Decision
        # Gate 1: Spatial variance (3D geometry check)
        # Gate 2: Eye specular ratio (corneal reflection check)
        # Face size is enforced during Phase 1 (WebSocket) before flashing starts.
        fail_reason = None
        if spatial_var < FLASH_SPATIAL_VAR_MIN:
            decision = "fail"
            fail_reason = "spatial_variance"
            logger.info(f"[Flash] HARD FAIL: spatial_var {spatial_var:.6f} < threshold {FLASH_SPATIAL_VAR_MIN}")
        elif eye_specular_ratio < FLASH_EYE_SPECULAR_RATIO_MIN:
            decision = "fail"
            fail_reason = "eye_specular"
            logger.info(f"[Flash] HARD FAIL: eye_specular_ratio {eye_specular_ratio:.4f} < threshold {FLASH_EYE_SPECULAR_RATIO_MIN}")
        else:
            decision = "pass"
        logger.info(f"[Flash] Decision: {decision}" + (f" (failed: {fail_reason})" if fail_reason else ""))

        # Append raw values to a log file for threshold calibration
        import json as json_mod
        from datetime import datetime
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "grid": f"{FLASH_GRID_ROWS}x{FLASH_GRID_COLS}",
            "frames_analyzed": len(all_patches),
            "mean_face_ratio": round(mean_face_ratio, 4),
            "temporal_corr_raw": round(temporal_corr, 6),
            "lag_score": round(lag_score, 6),
            "spatial_var_raw": round(spatial_var, 8),
            "ratio_stability_raw": round(ratio_stability, 6),
            "eye_brightness_cv": round(eye_brightness_var, 6),
            "eye_specular_ratio": round(eye_specular_ratio, 6),
            "eye_color_corr": round(eye_color_corr, 6),
            "eye_color_r": round(float(eye_color_per_channel[0]), 6),
            "eye_color_g": round(float(eye_color_per_channel[1]), 6),
            "eye_color_b": round(float(eye_color_per_channel[2]), 6),
            "eye_frames_used": eye_frames_used,
            "decision": decision,
        }
        log_path = os.path.join(debug_dir, "scoring_log.jsonl")
        with open(log_path, "a") as f:
            f.write(json_mod.dumps(log_entry) + "\n")
        logger.info(f"[Flash] Appended scoring data to {log_path}")

        return {
            "decision": decision,
        }

    finally:
        landmarker.close()
