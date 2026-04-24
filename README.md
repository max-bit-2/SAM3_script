# SAM3 Video Background Removal

Click on the object you want to keep, or drag a box around it, in the frame at about 1 second into the video. SAM3 tracks it through the entire video and outputs a clean `.mp4` with the background replaced by white or black.

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
pip install -e ./sam3 -r project_requirements.txt
```

### 6. Download model weights

Model weights require access approval via the SAM3 Hugging Face page.

Once approved, download a checkpoint (e.g. `sam3_large.pt`) and place it in the `weights/` directory:

```
weights/
└── sam3_large.pt
```

### 7. Notes on dependencies

`project_requirements.txt` is a curated runtime dependency list for this app and Docker image. It is not meant to be replaced with a machine-specific `pip freeze` output.

---

## Usage

```bash
python inference.py --video_path input.mp4
```

| Flag | Required | Description |
|------|----------|-------------|
| `--video_path` | Yes | Path to input video |
| `--output_path` | No | Output directory (default: `./outputs`, auto-named `<stem>_bg_removed.mp4`) |
| `--fps` | No | Output frame rate (default: native video FPS) |
| `--bg_color` | No | `white` or `black` (default: `white`) |
| `--checkpoint` | No | Specific checkpoint file to use. Defaults to `weights/sam3_large.pt` when present |
| `--model_version` | No | `auto`, `sam3`, or `sam3.1` (default: `auto`) |

**Controls in the prompt window:**
- Left-click to select a point on the object
- Left-click and drag to draw a box prompt
- **Enter** to confirm
- **Esc** to cancel

Prompt-frame behavior:
- The prompt UI opens on the frame at approximately 1 second into the source video.
- If the clip is shorter than 1 second, the last available frame is used.

---

## Docker

```bash
docker build -t sam3-bg-removal .
docker run --gpus all \
  -v /path/to/weights:/app/weights \
  -v /path/to/videos:/data \
  sam3-bg-removal \
  --video_path /data/input.mp4
```

---

## Notes

- `weights/` is git-ignored and must be populated manually.
- `outputs/` is git-ignored.
- `venv/` is git-ignored.
- If multiple checkpoints exist in `weights/`, the script prefers `sam3_large.pt` when present; otherwise pass `--checkpoint` explicitly.
- If tracking drops out briefly, the script now prefers temporal recovery: it carries forward recent valid masks, interpolates short gaps, and falls back to background-only output when the object is considered lost.
