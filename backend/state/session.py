from collections import deque
from enum import Enum
from dataclasses import dataclass, field

from config import CLEAR_FRAMES_REQUIRED


class VerificationStep(str, Enum):
    WAITING_FOR_FACE = "WAITING_FOR_FACE"
    CHECKING_OCCLUSION = "CHECKING_OCCLUSION"
    CHALLENGE = "CHALLENGE"
    CAPTURING = "CAPTURING"
    VERIFYING = "VERIFYING"
    RESULT = "RESULT"


class ChallengeStep(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    CHALLENGE_DONE = "CHALLENGE_DONE"


@dataclass
class SessionState:
    step: VerificationStep = VerificationStep.WAITING_FOR_FACE

    # Blink tracking
    left_blink_count: int = 0
    right_blink_count: int = 0
    left_closed_frames: int = 0
    right_closed_frames: int = 0
    mouth_open: bool = False

    # Occlusion stability
    consecutive_clear_frames: int = 0
    clear_frames_required: int = field(default_factory=lambda: CLEAR_FRAMES_REQUIRED)

    # Capture countdown
    capture_countdown: int = 0

    # Result
    spoof_label: int | None = None
    spoof_confidence: float | None = None

    def reset(self):
        self.step = VerificationStep.WAITING_FOR_FACE
        self.left_blink_count = 0
        self.right_blink_count = 0
        self.left_closed_frames = 0
        self.right_closed_frames = 0
        self.mouth_open = False
        self.consecutive_clear_frames = 0
        self.capture_countdown = 0
        self.spoof_label = None
        self.spoof_confidence = None


MOUTH_WINDOW = 10


@dataclass
class ActiveSessionState(SessionState):
    challenge_step: ChallengeStep = ChallengeStep.IN_PROGRESS
    mouth_open_start: float | None = None
    mouth_history: deque = field(default_factory=lambda: deque(maxlen=MOUTH_WINDOW))

    def reset(self):
        super().reset()
        self.challenge_step = ChallengeStep.IN_PROGRESS
        self.mouth_open_start = None
        self.mouth_history.clear()
