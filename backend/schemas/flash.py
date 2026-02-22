from pydantic import BaseModel


class FlashColorEntry(BaseModel):
    timestamp: float
    color: list[int]  # [R, G, B], each 0-255


class FlashFrameEntry(BaseModel):
    timestamp: float
    jpeg_b64: str  # base64-encoded JPEG


class FlashVerifyRequest(BaseModel):
    challenge_log: list[FlashColorEntry]
    frames: list[FlashFrameEntry]


class FlashVerifyResponse(BaseModel):
    type: str = "flash_result"
    decision: str  # "pass" | "fail"
