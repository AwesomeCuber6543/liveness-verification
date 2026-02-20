const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
const WS_BASE = BACKEND_URL.replace(/^http/, "ws");

export const WS_URL = `${WS_BASE}/ws/verify/passive`;
export const WS_URL_ACTIVE = `${WS_BASE}/ws/verify/active`;
export const FRAME_WIDTH = 640;
export const FRAME_HEIGHT = 480;
export const JPEG_QUALITY = 0.8;
