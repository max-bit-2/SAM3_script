FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY project_requirements.txt .
RUN pip install --no-cache-dir -r project_requirements.txt

COPY . .

# weights/ is mounted at runtime — do not bake into the image
VOLUME ["/app/weights"]

ENTRYPOINT ["python", "inference.py"]
