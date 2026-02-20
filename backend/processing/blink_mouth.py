from config import LEFT_EYE, RIGHT_EYE, EAR_THRESHOLD, EAR_CONSEC_FRAMES
from config import MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT, MAR_THRESHOLD


def _dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def compute_ear(landmarks, eye_indices):
    """Eye Aspect Ratio = (|p2-p6| + |p3-p5|) / (2*|p1-p4|)"""
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    vertical1 = _dist(p2, p6)
    vertical2 = _dist(p3, p5)
    horizontal = _dist(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


def compute_mar(landmarks):
    """Mouth Aspect Ratio = |top-bottom| / |left-right|"""
    vertical = _dist(landmarks[MOUTH_TOP], landmarks[MOUTH_BOTTOM])
    horizontal = _dist(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT])
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def update_blink_counts(landmarks, session):
    """Compute EAR for both eyes and update blink counters in session."""
    left_ear = compute_ear(landmarks, LEFT_EYE)
    right_ear = compute_ear(landmarks, RIGHT_EYE)

    if left_ear < EAR_THRESHOLD:
        session.left_closed_frames += 1
    else:
        if session.left_closed_frames >= EAR_CONSEC_FRAMES:
            session.left_blink_count += 1
        session.left_closed_frames = 0

    if right_ear < EAR_THRESHOLD:
        session.right_closed_frames += 1
    else:
        if session.right_closed_frames >= EAR_CONSEC_FRAMES:
            session.right_blink_count += 1
        session.right_closed_frames = 0

    mar = compute_mar(landmarks)
    session.mouth_open = mar > MAR_THRESHOLD

    return left_ear, right_ear, mar
