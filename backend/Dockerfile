FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download model weights from GitHub Release
RUN mkdir -p weights && \
    curl -L -o weights/2.7_80x80_MiniFASNetV2.pth \
      https://github.com/AwesomeCuber6543/liveness-verification/releases/download/v1.0-weights/2.7_80x80_MiniFASNetV2.pth && \
    curl -L -o weights/4_0_0_80x80_MiniFASNetV1SE.pth \
      https://github.com/AwesomeCuber6543/liveness-verification/releases/download/v1.0-weights/4_0_0_80x80_MiniFASNetV1SE.pth && \
    curl -L -o weights/best_convnext_tiny.pth \
      https://github.com/AwesomeCuber6543/liveness-verification/releases/download/v1.0-weights/best_convnext_tiny.pth && \
    curl -L -o weights/face_landmarker.task \
      https://github.com/AwesomeCuber6543/liveness-verification/releases/download/v1.0-weights/face_landmarker.task

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
