#!/usr/bin/env python3
import argparse
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import ffmpeg

from sam3.model_builder import build_sam3_video_model

MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 720
WINDOW = "SAM3 | Click object to keep — Enter to confirm, Esc to cancel"


def parse_args():
    p = argparse.ArgumentParser(description="SAM3 video background removal")
    p.add_argument("--video_path", required=True, help="Path to input video")
    p.add_argument("--output_path", required=True, help="Output directory")
    p.add_argument("--fps", type=float, default=None, help="Output FPS (default: native)")
    p.add_argument("--bg_color", choices=["white", "black"], default="white")
    p.add_argument("--prompt", type=str, default=None, help="(stub) Text prompt — not yet implemented")
    return p.parse_args()


def open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Error: Cannot open video '{path}'")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, w, h, n


def collect_click(frame: np.ndarray, native_w: int, native_h: int):
    """Show frame, let user click + confirm. Returns native (x, y) or None."""
    scale = min(MAX_DISPLAY_W / native_w, MAX_DISPLAY_H / native_h, 1.0)
    disp_w, disp_h = int(native_w * scale), int(native_h * scale)
    base = cv2.resize(frame, (disp_w, disp_h))
    state = {"pt": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pt"] = (x, y)
            tmp = base.copy()
            cv2.circle(tmp, (x, y), 8, (0, 255, 0), -1)
            cv2.imshow(WINDOW, tmp)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    cv2.imshow(WINDOW, base)

    confirmed = False
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and state["pt"] is not None:  # Enter
            confirmed = True
            break
        if key == 27:  # Esc
            break

    cv2.destroyAllWindows()
    if not confirmed:
        return None
    cx, cy = state["pt"]
    return cx / scale, cy / scale  # scale back to native coords


def load_predictor(weights_dir: Path, device: str):
    checkpoints = sorted(weights_dir.glob("*.pt")) + sorted(weights_dir.glob("*.pth"))
    if not checkpoints:
        sys.exit(
            f"Error: No model weights found in '{weights_dir}'.\n"
            "Download a SAM3 checkpoint into the weights/ directory.\n"
            "See README.md for instructions."
        )
    ckpt = str(checkpoints[0])
    print(f"Loading weights: {ckpt}")
    bpe_path = str(
        Path(__file__).parent / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    )
    model = build_sam3_video_model(
        checkpoint_path=ckpt,
        load_from_HF=False,
        bpe_path=bpe_path,
        device=device,
    )
    # The SAM2-compatible tracker API is on model.tracker;
    # it needs the detector backbone for visual feature extraction.
    predictor = model.tracker
    predictor.backbone = model.detector.backbone
    return predictor


def extract_mask(video_res_masks, idx: int = 0) -> np.ndarray | None:
    """Return (H, W) bool numpy mask for object at index idx, or None."""
    if video_res_masks is None or len(video_res_masks) <= idx:
        return None
    m = (video_res_masks[idx] > 0.0).cpu().numpy()
    if m.ndim == 3:   # (1, H, W) → (H, W)
        m = m[0]
    return m.astype(bool)


def composite(frame: np.ndarray, mask: np.ndarray | None, bg_value: int) -> np.ndarray:
    out = frame.copy()
    if mask is None or not mask.any():
        warnings.warn("Empty mask on frame — writing frame without compositing")
        return out
    out[~mask] = bg_value
    return out


def merge_audio(tmp_file: Path, source_video: str, output_file: Path):
    try:
        (
            ffmpeg
            .output(
                ffmpeg.input(str(tmp_file)).video,
                ffmpeg.input(source_video).audio,
                str(output_file),
                vcodec="copy",
                acodec="aac",
            )
            .overwrite_output()
            .run(quiet=True)
        )
        tmp_file.unlink()
    except ffmpeg.Error as e:
        stderr = e.stderr.decode() if e.stderr else str(e)
        warnings.warn(f"Audio passthrough failed — saving without audio. ({stderr})")
        tmp_file.rename(output_file)


def main():
    args = parse_args()

    if args.prompt is not None:
        print("Warning: --prompt is not yet implemented. Using click mode.")

    if not os.path.exists(args.video_path):
        sys.exit(f"Error: Video file not found: '{args.video_path}'")

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video_path).stem
    output_file = output_dir / f"{stem}_bg_removed.mp4"
    tmp_file = output_dir / f"{stem}.noaudio.tmp.mp4"

    bg_value = 255 if args.bg_color == "white" else 0

    cap, native_fps, width, height, total_frames = open_video(args.video_path)
    prompt_frame_idx = round(native_fps)  # 1 second in
    cap.set(cv2.CAP_PROP_POS_FRAMES, prompt_frame_idx)
    ret, prompt_frame = cap.read()
    cap.release()
    if not ret:
        sys.exit("Error: Could not read frame at 1 second.")

    click = collect_click(prompt_frame, width, height)
    if click is None:
        print("Cancelled.")
        sys.exit(0)

    native_x, native_y = click
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    weights_dir = Path(__file__).parent / "weights"
    predictor = load_predictor(weights_dir, device)

    out_fps = args.fps if (args.fps and args.fps < native_fps) else native_fps
    frame_step = round(native_fps / out_fps) if out_fps < native_fps else 1

    # Normalize click coords to [0, 1] as required by SAM3 tracker
    pts = torch.tensor([[native_x / width, native_y / height]], dtype=torch.float32)
    lbl = torch.tensor([1], dtype=torch.int32)  # 1 = foreground click

    inference_state = predictor.init_state(video_path=args.video_path)
    predictor.clear_all_points_in_video(inference_state)

    predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=prompt_frame_idx,
        obj_id=1,
        points=pts,
        labels=lbl,
        clear_old_points=False,
    )

    # Collect all masks: forward from prompt frame, then backward to frame 0
    masks_by_frame: dict[int, np.ndarray] = {}

    print(f"Propagating forward from frame {prompt_frame_idx}...")
    for f_idx, obj_ids, _, video_res_masks, _ in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=prompt_frame_idx,
        max_frame_num_to_track=total_frames - prompt_frame_idx,
        reverse=False,
        propagate_preflight=True,
    ):
        if len(obj_ids) > 0:
            mask = extract_mask(video_res_masks, idx=0)
            if mask is not None:
                masks_by_frame[f_idx] = mask
        if f_idx % 100 == 0:
            print(f"  Frame {f_idx} / {total_frames}")

    if prompt_frame_idx > 0:
        print(f"Propagating backward from frame {prompt_frame_idx - 1}...")
        for f_idx, obj_ids, _, video_res_masks, _ in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=prompt_frame_idx - 1,
            max_frame_num_to_track=prompt_frame_idx,
            reverse=True,
            propagate_preflight=True,
        ):
            if len(obj_ids) > 0:
                mask = extract_mask(video_res_masks, idx=0)
                if mask is not None:
                    masks_by_frame[f_idx] = mask
            if f_idx % 100 == 0:
                print(f"  Frame {f_idx} / {total_frames}")

    print(f"Collected masks for {len(masks_by_frame)} / {total_frames} frames.")

    # Pipe composited frames to ffmpeg
    ffmpeg_proc = (
        ffmpeg
        .input("pipe:", format="rawvideo", pix_fmt="bgr24",
               s=f"{width}x{height}", r=out_fps)
        .output(str(tmp_file), pix_fmt="yuv420p", vcodec="libx264", crf=18)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    cap = cv2.VideoCapture(args.video_path)
    written = 0
    print("Compositing and writing output...")
    for f_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_step > 1 and f_idx % frame_step != 0:
            continue
        mask = masks_by_frame.get(f_idx)
        out_frame = composite(frame, mask, bg_value)
        ffmpeg_proc.stdin.write(out_frame.tobytes())
        written += 1

    cap.release()
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()

    print(f"Composited {written} frames. Merging audio...")
    merge_audio(tmp_file, args.video_path, output_file)
    print(f"Done. Output: {output_file}")


if __name__ == "__main__":
    main()
