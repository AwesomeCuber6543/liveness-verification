from enum import Enum
from dataclasses import dataclass, field

from config import CLEAR_FRAMES_REQUIRED


class FlashStep(str, Enum):
    WAITING_FOR_FACE = "WAITING_FOR_FACE"
    CHECKING_OCCLUSION = "CHECKING_OCCLUSION"
    READY_FOR_FLASH = "READY_FOR_FLASH"


@dataclass
class FlashSessionState:
    step: FlashStep = FlashStep.WAITING_FOR_FACE

    # Blink tracking (kept for compatibility with update_blink_counts)
    left_blink_count: int = 0
    right_blink_count: int = 0
    left_closed_frames: int = 0
    right_closed_frames: int = 0
    mouth_open: bool = False

    # Occlusion stability
    consecutive_clear_frames: int = 0
    clear_frames_required: int = field(default_factory=lambda: CLEAR_FRAMES_REQUIRED)

    def reset(self):
        self.step = FlashStep.WAITING_FOR_FACE
        self.left_blink_count = 0
        self.right_blink_count = 0
        self.left_closed_frames = 0
        self.right_closed_frames = 0
        self.mouth_open = False
        self.consecutive_clear_frames = 0
