# SAM3 Video Background Removal

Click on the object you want to keep in the first frame. SAM3 tracks it through the entire video and outputs a clean `.mp4` with the background replaced by white or black.

---

## Setup

### 1. Prerequisites

- Python 3.12 ([download](https://www.python.org/downloads/release/python-3120/))
- CUDA 12.6+ and a CUDA-capable GPU
- Git

### 2. Clone with submodules

```bash
git clone https://github.com/max-bit-2/SAM3_script.git
cd SAM3_script
git submodule update --init --recursive
```

### 3. Create virtual environment

```bash
# Windows
py -3.12 -m venv venv
venv\Scripts\activate

# macOS / Linux
python3.12 -m venv venv
source venv/bin/activate
```

### 4. Install PyTorch (CUDA 12.8)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

> If `cu128` is unavailable, use `cu124` — it is forward-compatible with CUDA 12.8 drivers.

### 5. Install dependencies

```bash
pip install -r sam3/requirements.txt
pip install opencv-python ffmpeg-python
```

### 6. Download model weights

Model weights require access approval via the SAM3 Hugging Face page.

Once approved, download a checkpoint (e.g. `sam3_large.pt`) and place it in the `weights/` directory:

```
weights/
└── sam3_large.pt
```

### 7. Freeze dependencies (after confirming end-to-end)

```bash
pip freeze > project_requirements.txt
```

---

## Usage

```bash
python inference.py --video_path input.mp4 --output_path outputs/
```

| Flag | Required | Description |
|------|----------|-------------|
| `--video_path` | Yes | Path to input video |
| `--output_path` | Yes | Output directory (auto-named `<stem>_bg_removed.mp4`) |
| `--fps` | No | Output frame rate (default: native video FPS) |
| `--bg_color` | No | `white` or `black` (default: `white`) |

**Controls in the click window:**
- Left-click to select a point on the object (re-click to reposition)
- **Enter** to confirm
- **Esc** to cancel

---

## Docker

```bash
docker build -t sam3-bg-removal .
docker run --gpus all \
  -v /path/to/weights:/app/weights \
  -v /path/to/videos:/data \
  sam3-bg-removal \
  --video_path /data/input.mp4 \
  --output_path /data/outputs/
```

---

## Notes

- `weights/` is git-ignored and must be populated manually.
- `outputs/` is git-ignored.
- `venv/` is git-ignored.
