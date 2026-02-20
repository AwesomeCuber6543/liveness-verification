import asyncio
import json
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from models.loader import load_all_models
from processing.face_detection import create_landmarker
from processing.pipeline import process_frame, process_frame_active
from state.session import SessionState, ActiveSessionState

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    app.state.registry = load_all_models()
    print("All models loaded. Server ready.")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": len(app.state.registry.antispoof_models)}


@app.websocket("/ws/verify/passive")
async def passive_verification(websocket: WebSocket):
    await websocket.accept()
    session = SessionState()
    registry = websocket.app.state.registry
    landmarker = create_landmarker()
    frame_count = 0
    latest_frame_bytes: bytes | None = None

    logger.info("WS session started, creating landmarker")

    async def reader():
        """Continuously read from WebSocket, keeping only the latest binary frame."""
        nonlocal latest_frame_bytes
        try:
            while True:
                message = await websocket.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                if "text" in message:
                    try:
                        data = json.loads(message["text"])
                        if data.get("type") == "reset":
                            logger.info("WS reset command received")
                            session.reset()
                            latest_frame_bytes = None
                            await websocket.send_json({"type": "reset_ack", "step": session.step.value})
                    except json.JSONDecodeError:
                        pass

                if "bytes" in message:
                    # Always overwrite — we only care about the latest frame
                    latest_frame_bytes = message["bytes"]

        except (WebSocketDisconnect, RuntimeError):
            pass

    async def processor():
        """Process the latest frame, skipping stale ones."""
        nonlocal latest_frame_bytes, frame_count
        try:
            while True:
                if latest_frame_bytes is None:
                    await asyncio.sleep(0.01)
                    continue

                # Grab and clear the latest frame
                jpeg_bytes = latest_frame_bytes
                latest_frame_bytes = None

                frame = cv2.imdecode(
                    np.frombuffer(jpeg_bytes, np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if frame is None:
                    await websocket.send_json({"type": "error", "message": "Could not decode frame"})
                    continue

                frame_count += 1
                result = await asyncio.to_thread(
                    process_frame, frame, registry, session, landmarker
                )

                if frame_count <= 3 or frame_count % 30 == 0:
                    logger.info(f"WS frame #{frame_count} -> step={result.get('step')}, face={result.get('face_detected')}")

                try:
                    await websocket.send_json(result)
                except (WebSocketDisconnect, RuntimeError):
                    break

        except (WebSocketDisconnect, RuntimeError):
            pass
        except asyncio.CancelledError:
            pass

    try:
        # Run reader and processor concurrently
        reader_task = asyncio.create_task(reader())
        processor_task = asyncio.create_task(processor())

        # When reader finishes (disconnect), cancel processor
        await reader_task
        processor_task.cancel()
        try:
            await processor_task
        except asyncio.CancelledError:
            pass

    except (WebSocketDisconnect, RuntimeError) as e:
        logger.info(f"WS session ended: {type(e).__name__}: {e}")
    finally:
        logger.info(f"WS cleanup: processed {frame_count} frames, closing landmarker")
        landmarker.close()


@app.websocket("/ws/verify/active")
async def active_verification(websocket: WebSocket):
    await websocket.accept()
    session = ActiveSessionState()
    registry = websocket.app.state.registry
    landmarker = create_landmarker()
    frame_count = 0
    latest_frame_bytes: bytes | None = None

    logger.info("WS active session started, creating landmarker")

    async def reader():
        nonlocal latest_frame_bytes
        try:
            while True:
                message = await websocket.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                if "text" in message:
                    try:
                        data = json.loads(message["text"])
                        if data.get("type") == "reset":
                            logger.info("WS active reset command received")
                            session.reset()
                            latest_frame_bytes = None
                            await websocket.send_json({"type": "reset_ack", "step": session.step.value})
                    except json.JSONDecodeError:
                        pass

                if "bytes" in message:
                    latest_frame_bytes = message["bytes"]

        except (WebSocketDisconnect, RuntimeError):
            pass

    async def processor():
        nonlocal latest_frame_bytes, frame_count
        try:
            while True:
                if latest_frame_bytes is None:
                    await asyncio.sleep(0.01)
                    continue

                jpeg_bytes = latest_frame_bytes
                latest_frame_bytes = None

                frame = cv2.imdecode(
                    np.frombuffer(jpeg_bytes, np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if frame is None:
                    await websocket.send_json({"type": "error", "message": "Could not decode frame"})
                    continue

                frame_count += 1
                result = await asyncio.to_thread(
                    process_frame_active, frame, registry, session, landmarker
                )

                logger.info(f"WS active frame #{frame_count} -> step={result.get('step')}, challenge={result.get('challenge')}")

                try:
                    await websocket.send_json(result)
                except (WebSocketDisconnect, RuntimeError):
                    break

        except (WebSocketDisconnect, RuntimeError):
            pass
        except asyncio.CancelledError:
            pass

    try:
        reader_task = asyncio.create_task(reader())
        processor_task = asyncio.create_task(processor())

        await reader_task
        processor_task.cancel()
        try:
            await processor_task
        except asyncio.CancelledError:
            pass

    except (WebSocketDisconnect, RuntimeError) as e:
        logger.info(f"WS active session ended: {type(e).__name__}: {e}")
    finally:
        logger.info(f"WS active cleanup: processed {frame_count} frames, closing landmarker")
        landmarker.close()
