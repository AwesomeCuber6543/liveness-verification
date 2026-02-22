"""
Debug script: Extract eye crops from saved flash frames.

Usage:
  1. Run a flash verification (real face or phone photo)
  2. The flash scoring endpoint will save frames to debug_flash_frames/
  3. Run this script: python3 debug_eye_crops.py

It will extract left and right eye crops from a few random frames
and save them to debug_eye_crops/ for visual inspection.
"""

import os
import random
import cv2
import numpy as np
from processing.face_detection import create_landmarker, detect_face

# MediaPipe eye landmark indices for tight eye region crops
# These form a bounding region around each eye
LEFT_EYE_REGION = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_REGION = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Iris landmarks (MediaPipe 468+ model provides these at indices 468-477)
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def get_eye_bbox(landmarks, eye_indices, img_w, img_h, padding=10):
    """Get a bounding box around an eye from landmark indices."""
    xs = [landmarks[i].x * img_w for i in eye_indices]
    ys = [landmarks[i].y * img_h for i in eye_indices]

    x_min = max(0, int(min(xs)) - padding)
    y_min = max(0, int(min(ys)) - padding)
    x_max = min(img_w, int(max(xs)) + padding)
    y_max = min(img_h, int(max(ys)) + padding)

    return x_min, y_min, x_max, y_max


def extract_eyes_from_frame(frame_bgr, landmarker, frame_idx, output_dir):
    """Extract left and right eye crops from a frame."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    landmarks, bbox = detect_face(landmarker, frame_rgb)

    if landmarks is None:
        print(f"  Frame {frame_idx}: no face detected, skipping")
        return None

    img_h, img_w = frame_bgr.shape[:2]

    # Extract left eye
    lx1, ly1, lx2, ly2 = get_eye_bbox(landmarks, LEFT_EYE_REGION, img_w, img_h, padding=15)
    left_eye = frame_bgr[ly1:ly2, lx1:lx2]

    # Extract right eye
    rx1, ry1, rx2, ry2 = get_eye_bbox(landmarks, RIGHT_EYE_REGION, img_w, img_h, padding=15)
    right_eye = frame_bgr[ry1:ry2, rx1:rx2]

    # Upscale for better visibility (4x)
    if left_eye.size > 0:
        left_eye_big = cv2.resize(left_eye, (left_eye.shape[1] * 4, left_eye.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(output_dir, f"frame{frame_idx:03d}_left_eye.png"), left_eye_big)

    if right_eye.size > 0:
        right_eye_big = cv2.resize(right_eye, (right_eye.shape[1] * 4, right_eye.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(output_dir, f"frame{frame_idx:03d}_right_eye.png"), right_eye_big)

    # Also compute mean RGB of eye regions
    left_mean = left_eye.mean(axis=(0, 1)) if left_eye.size > 0 else [0, 0, 0]
    right_mean = right_eye.mean(axis=(0, 1)) if right_eye.size > 0 else [0, 0, 0]

    # BGR to RGB for display
    print(f"  Frame {frame_idx}: left_eye_rgb=({left_mean[2]:.0f},{left_mean[1]:.0f},{left_mean[0]:.0f}), "
          f"right_eye_rgb=({right_mean[2]:.0f},{right_mean[1]:.0f},{right_mean[0]:.0f})")

    return left_mean, right_mean


def main():
    frames_dir = "debug_flash_frames"
    output_dir = "debug_eye_crops"

    if not os.path.exists(frames_dir):
        print(f"No frames found at {frames_dir}/")
        print("Run a flash verification first — frames will be saved automatically.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # List all saved frames
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png")])

    if not frame_files:
        print(f"No frame images found in {frames_dir}/")
        return

    print(f"Found {len(frame_files)} frames")

    # Pick ~8 evenly spaced frames (or all if fewer)
    num_samples = min(8, len(frame_files))
    indices = [int(i * len(frame_files) / num_samples) for i in range(num_samples)]
    sampled = [frame_files[i] for i in indices]

    print(f"Extracting eye crops from {num_samples} frames...")

    landmarker = create_landmarker()

    try:
        for fname in sampled:
            frame_bgr = cv2.imread(os.path.join(frames_dir, fname))
            if frame_bgr is None:
                print(f"  Could not read {fname}")
                continue

            idx = int(fname.split("_")[1].split(".")[0]) if "_" in fname else 0
            extract_eyes_from_frame(frame_bgr, landmarker, idx, output_dir)
    finally:
        landmarker.close()

    print(f"\nEye crops saved to {output_dir}/")
    print("Compare real face vs phone photo eye crops to see color reflection differences.")


if __name__ == "__main__":
    main()
