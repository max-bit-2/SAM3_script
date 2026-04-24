FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir -e ./sam3 -r project_requirements.txt

# weights/ is mounted at runtime - do not bake into the image
VOLUME ["/app/weights"]

ENTRYPOINT ["python", "inference.py"]
