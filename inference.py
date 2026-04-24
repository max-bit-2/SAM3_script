#!/usr/bin/env python3
import argparse
import gc
import importlib.util
import math
import os
import sys
import tempfile
import warnings
from pathlib import Path

import cv2
import ffmpeg
import numpy as np
import torch

from sam3.model_builder import build_sam3_predictor, build_sam3_video_model

MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 720
WINDOW = "SAM3 | Click for point, drag for box - Enter to confirm, Esc to cancel"
DEFAULT_OBJECT_ID = 1
MASK_STATE_MISSING = 0
MASK_STATE_VALID = 1
MASK_STATE_EMPTY = 2
BOX_DRAG_THRESHOLD_PX = 8

FALLBACK_STATE_VALID = "valid"
FALLBACK_STATE_LOW_CONFIDENCE = "low_confidence"
FALLBACK_STATE_EMPTY = "empty"
FALLBACK_STATE_MISSING = "missing"
FALLBACK_STATE_CARRIED = "carried"
FALLBACK_STATE_INTERPOLATED = "interpolated"
FALLBACK_STATE_LOST = "lost"


def parse_args():
    p = argparse.ArgumentParser(description="SAM3 video background removal")
    p.add_argument("--video_path", required=True, help="Path to input video")
    p.add_argument(
        "--output_path",
        default="outputs",
        help="Output directory (default: ./outputs)",
    )
    p.add_argument("--fps", type=float, default=None, help="Output FPS (default: native)")
    p.add_argument("--bg_color", choices=["white", "black"], default="white")
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="(stub) Text prompt - not yet implemented",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file to use. If omitted, exactly one checkpoint must exist in weights/",
    )
    p.add_argument(
        "--model_version",
        choices=["auto", "sam3", "sam3.1"],
        default="auto",
        help="Predictor version. 'auto' infers from the checkpoint name when possible.",
    )
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


def collect_prompt(frame: np.ndarray, native_w: int, native_h: int):
    """Show frame, let user choose a point or drag a box, then confirm."""
    scale = min(MAX_DISPLAY_W / native_w, MAX_DISPLAY_H / native_h, 1.0)
    disp_w, disp_h = int(native_w * scale), int(native_h * scale)
    base = cv2.resize(frame, (disp_w, disp_h))
    state = {
        "mouse_down": False,
        "start_pt": None,
        "current_pt": None,
        "prompt": None,
    }

    def render_preview():
        tmp = base.copy()
        prompt = state["prompt"]
        if prompt is not None:
            if prompt["type"] == "point":
                cv2.circle(tmp, prompt["point"], 8, (0, 255, 0), -1)
            elif prompt["type"] == "box":
                x1, y1 = prompt["start_pt"]
                x2, y2 = prompt["end_pt"]
                cv2.rectangle(tmp, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if state["mouse_down"] and state["start_pt"] is not None and state["current_pt"] is not None:
            x1, y1 = state["start_pt"]
            x2, y2 = state["current_pt"]
            cv2.rectangle(tmp, (x1, y1), (x2, y2), (0, 200, 255), 1)
        cv2.imshow(WINDOW, tmp)

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            state["mouse_down"] = True
            state["start_pt"] = (x, y)
            state["current_pt"] = (x, y)
            render_preview()
        elif event == cv2.EVENT_MOUSEMOVE and state["mouse_down"]:
            state["current_pt"] = (x, y)
            render_preview()
        elif event == cv2.EVENT_LBUTTONUP and state["mouse_down"]:
            state["mouse_down"] = False
            state["current_pt"] = (x, y)

            start_x, start_y = state["start_pt"]
            end_x, end_y = state["current_pt"]
            dx = abs(end_x - start_x)
            dy = abs(end_y - start_y)

            if dx >= BOX_DRAG_THRESHOLD_PX or dy >= BOX_DRAG_THRESHOLD_PX:
                x1, x2 = sorted((start_x, end_x))
                y1, y2 = sorted((start_y, end_y))
                state["prompt"] = {
                    "type": "box",
                    "start_pt": (x1, y1),
                    "end_pt": (x2, y2),
                }
            else:
                state["prompt"] = {
                    "type": "point",
                    "point": (end_x, end_y),
                }
            render_preview()

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    cv2.imshow(WINDOW, base)

    confirmed = False
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and state["prompt"] is not None:  # Enter
            confirmed = True
            break
        if key == 27:  # Esc
            break

    cv2.destroyAllWindows()
    if not confirmed:
        return None

    prompt = state["prompt"]
    if prompt["type"] == "point":
        cx, cy = prompt["point"]
        return {
            "type": "point",
            "point": (cx / scale, cy / scale),
        }

    x1, y1 = prompt["start_pt"]
    x2, y2 = prompt["end_pt"]
    return {
        "type": "box",
        "box_xywh": (
            x1 / scale,
            y1 / scale,
            (x2 - x1) / scale,
            (y2 - y1) / scale,
        ),
    }


def infer_model_version(checkpoint_path: Path) -> str:
    name = checkpoint_path.name.lower()
    if "sam3.1" in name or "sam3_1" in name or "multiplex" in name:
        return "sam3.1"
    if "sam3" in name:
        return "sam3"
    raise SystemExit(
        "Error: Could not infer --model_version from checkpoint name "
        f"'{checkpoint_path.name}'. Pass --model_version explicitly."
    )


def resolve_checkpoint(args, weights_dir: Path) -> tuple[Path, str]:
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            sys.exit(f"Error: Checkpoint not found: '{checkpoint_path}'")
    else:
        checkpoints = sorted(weights_dir.glob("*.pt")) + sorted(weights_dir.glob("*.pth"))
        if not checkpoints:
            sys.exit(
                f"Error: No model weights found in '{weights_dir}'.\n"
                "Download a SAM3 checkpoint into the weights/ directory.\n"
                "See README.md for instructions."
            )
        preferred_names = ("sam3_large.pt", "sam3_large.pth")
        preferred_matches = [
            checkpoint.resolve()
            for checkpoint in checkpoints
            if checkpoint.name.lower() in preferred_names
        ]
        if len(preferred_matches) == 1:
            checkpoint_path = preferred_matches[0]
        elif len(preferred_matches) > 1:
            ckpt_list = "\n".join(f"  - {p.name}" for p in preferred_matches)
            sys.exit(
                "Error: Multiple preferred SAM3 large checkpoints found. Pass --checkpoint "
                f"explicitly.\n{ckpt_list}"
            )
        elif len(checkpoints) == 1:
            checkpoint_path = checkpoints[0].resolve()
        else:
            ckpt_list = "\n".join(f"  - {p.name}" for p in checkpoints)
            sys.exit(
                "Error: Multiple checkpoints found and no default 'sam3_large.pt' was present. "
                f"Pass --checkpoint explicitly.\n{ckpt_list}"
            )

    model_version = (
        infer_model_version(checkpoint_path)
        if args.model_version == "auto"
        else args.model_version
    )
    return checkpoint_path, model_version


def pick_video_loader(model_version: str) -> str | None:
    if model_version != "sam3":
        return None
    return "torchcodec" if importlib.util.find_spec("torchcodec") is not None else "cv2"


def extract_tracker_mask(
    video_res_masks,
    obj_ids,
    obj_id: int = DEFAULT_OBJECT_ID,
) -> np.ndarray | None:
    if video_res_masks is None or obj_ids is None:
        return None

    for idx, current_obj_id in enumerate(obj_ids):
        if int(current_obj_id) == obj_id:
            mask = video_res_masks[idx]
            if isinstance(mask, torch.Tensor):
                mask = (mask > 0.0).cpu().numpy()
            else:
                mask = np.asarray(mask) > 0.0
            if mask.ndim == 3:
                mask = mask[0]
            return np.asarray(mask, dtype=bool)
    return None


class Sam3AppBackend:
    def start_session(self, video_path: str):
        raise NotImplementedError

    def add_visual_prompt(self, prompt_input, frame_idx: int, width: int, height: int):
        raise NotImplementedError

    def propagate_masks(self, prompt_frame_idx: int, total_frames: int):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class SharedPredictorBackend(Sam3AppBackend):
    def __init__(self, checkpoint_path: Path, model_version: str):
        build_kwargs = {
            "checkpoint_path": str(checkpoint_path),
            "version": model_version,
            "async_loading_frames": True,
        }
        video_loader_type = pick_video_loader(model_version)
        if video_loader_type is not None:
            build_kwargs["video_loader_type"] = video_loader_type

        self.predictor = build_sam3_predictor(**build_kwargs)
        self.video_loader_type = video_loader_type
        self.session_id = None

    def start_session(self, video_path: str):
        response = self.predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": video_path,
                "offload_video_to_cpu": True,
            }
        )
        self.session_id = response["session_id"]

    def add_visual_prompt(self, prompt_input, frame_idx: int, width: int, height: int):
        request = {
            "type": "add_prompt",
            "session_id": self.session_id,
            "frame_index": frame_idx,
            "obj_id": DEFAULT_OBJECT_ID,
        }
        if prompt_input["type"] == "point":
            request["points"] = [[prompt_input["point"][0] / width, prompt_input["point"][1] / height]]
            request["point_labels"] = [1]
        else:
            request["bounding_boxes"] = [[
                prompt_input["box_xywh"][0] / width,
                prompt_input["box_xywh"][1] / height,
                prompt_input["box_xywh"][2] / width,
                prompt_input["box_xywh"][3] / height,
            ]]
            request["bounding_box_labels"] = [1]
        self.predictor.handle_request(request)

    def propagate_masks(self, prompt_frame_idx: int, total_frames: int):
        for item in self.predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": self.session_id,
                "propagation_direction": "forward",
                "start_frame_index": prompt_frame_idx,
                "max_frame_num_to_track": total_frames - prompt_frame_idx,
            }
        ):
            yield item["frame_index"], extract_mask(item["outputs"], obj_id=DEFAULT_OBJECT_ID)

        if prompt_frame_idx > 0:
            for item in self.predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": self.session_id,
                    "propagation_direction": "backward",
                    "start_frame_index": prompt_frame_idx - 1,
                    "max_frame_num_to_track": prompt_frame_idx,
                }
            ):
                yield item["frame_index"], extract_mask(item["outputs"], obj_id=DEFAULT_OBJECT_ID)

    def close(self):
        if self.session_id is not None:
            self.predictor.handle_request({"type": "close_session", "session_id": self.session_id})
            self.session_id = None
        if hasattr(self.predictor, "shutdown"):
            self.predictor.shutdown()


class TrackerPredictorBackend(Sam3AppBackend):
    def __init__(self, checkpoint_path: Path):
        self.model = build_sam3_video_model(
            checkpoint_path=str(checkpoint_path),
            load_from_HF=False,
            device="cuda",
        )
        self.predictor = self.model.tracker
        self.predictor.backbone = self.model.detector.backbone
        self.video_loader_type = None
        self.inference_state = None

    def start_session(self, video_path: str):
        video_path_obj = Path(video_path)
        is_supported = video_path_obj.is_dir() or video_path_obj.suffix.lower() == ".mp4"
        if not is_supported:
            raise RuntimeError(
                "Base SAM3 interactive point/box prompting uses the tracker backend here, "
                "which currently supports MP4 files or frame directories. "
                f"Received: '{video_path}'"
            )
        self.inference_state = self.predictor.init_state(
            video_path=video_path,
            offload_video_to_cpu=True,
            async_loading_frames=True,
        )
        self.predictor.clear_all_points_in_video(self.inference_state)

    def add_visual_prompt(self, prompt_input, frame_idx: int, width: int, height: int):
        kwargs = {
            "inference_state": self.inference_state,
            "frame_idx": frame_idx,
            "obj_id": DEFAULT_OBJECT_ID,
            "clear_old_points": True,
        }
        if prompt_input["type"] == "point":
            kwargs["points"] = [[prompt_input["point"][0] / width, prompt_input["point"][1] / height]]
            kwargs["labels"] = [1]
        else:
            x, y, w, h = prompt_input["box_xywh"]
            kwargs["box"] = torch.tensor(
                [
                    x / width,
                    y / height,
                    (x + w) / width,
                    (y + h) / height,
                ],
                dtype=torch.float32,
            )
        self.predictor.add_new_points_or_box(**kwargs)

    def propagate_masks(self, prompt_frame_idx: int, total_frames: int):
        for frame_idx, obj_ids, _, video_res_masks, _ in self.predictor.propagate_in_video(
            self.inference_state,
            start_frame_idx=prompt_frame_idx,
            max_frame_num_to_track=total_frames - prompt_frame_idx,
            reverse=False,
            propagate_preflight=True,
        ):
            yield frame_idx, extract_tracker_mask(video_res_masks, obj_ids, obj_id=DEFAULT_OBJECT_ID)

        if prompt_frame_idx > 0:
            for frame_idx, obj_ids, _, video_res_masks, _ in self.predictor.propagate_in_video(
                self.inference_state,
                start_frame_idx=prompt_frame_idx - 1,
                max_frame_num_to_track=prompt_frame_idx,
                reverse=True,
                propagate_preflight=True,
            ):
                yield frame_idx, extract_tracker_mask(video_res_masks, obj_ids, obj_id=DEFAULT_OBJECT_ID)

    def close(self):
        self.inference_state = None
        self.predictor = None
        self.model = None


def load_backend(checkpoint_path: Path, model_version: str):
    if model_version == "sam3":
        backend = TrackerPredictorBackend(checkpoint_path)
        backend_name = "tracker"
    else:
        backend = SharedPredictorBackend(checkpoint_path, model_version)
        backend_name = "shared_predictor"
    return backend, backend_name


def extract_mask(outputs: dict | None, obj_id: int = DEFAULT_OBJECT_ID) -> np.ndarray | None:
    """Return the binary mask for obj_id as an (H, W) bool array, or None."""
    if not outputs:
        return None

    out_obj_ids = outputs.get("out_obj_ids")
    out_binary_masks = outputs.get("out_binary_masks")
    if out_obj_ids is None or out_binary_masks is None:
        return None

    if len(out_obj_ids) == 0:
        return None

    for idx, current_obj_id in enumerate(out_obj_ids):
        if int(current_obj_id) == obj_id:
            mask = np.asarray(out_binary_masks[idx], dtype=bool)
            if mask.ndim == 3:
                mask = mask[0]
            return mask
    return None


class PackedMaskStore:
    """Persist masks on disk so long videos do not require full in-memory storage."""

    def __init__(self, root_dir: Path, total_frames: int, height: int, width: int):
        self.height = height
        self.width = width
        self.total_frames = total_frames
        self.pixel_count = height * width
        self.bytes_per_frame = math.ceil(self.pixel_count / 8)

        self.mask_path = root_dir / "masks.bin"
        self.state_path = root_dir / "masks.state.bin"
        self.masks = np.memmap(
            self.mask_path,
            dtype=np.uint8,
            mode="w+",
            shape=(total_frames, self.bytes_per_frame),
        )
        self.state = np.memmap(
            self.state_path,
            dtype=np.uint8,
            mode="w+",
            shape=(total_frames,),
        )
        self.masks[:] = 0
        self.state[:] = MASK_STATE_MISSING

    def put(self, frame_idx: int, mask: np.ndarray):
        mask_bool = np.asarray(mask, dtype=bool).reshape(-1)
        if mask_bool.size != self.pixel_count:
            raise ValueError(
                f"Mask shape mismatch: expected {self.height}x{self.width}, got {mask.shape}"
            )
        packed = np.packbits(mask_bool, bitorder="little")
        self.masks[frame_idx, : packed.size] = packed
        if packed.size < self.bytes_per_frame:
            self.masks[frame_idx, packed.size :] = 0
        self.state[frame_idx] = MASK_STATE_VALID if mask_bool.any() else MASK_STATE_EMPTY

    def get(self, frame_idx: int) -> np.ndarray | None:
        if int(self.state[frame_idx]) == MASK_STATE_MISSING:
            return None
        packed = np.asarray(self.masks[frame_idx])
        unpacked = np.unpackbits(
            packed,
            bitorder="little",
            count=self.pixel_count,
        )
        return unpacked.reshape(self.height, self.width).astype(bool, copy=False)

    def flush(self):
        self.masks.flush()
        self.state.flush()

    def get_state(self, frame_idx: int) -> int:
        return int(self.state[frame_idx])

    def close(self):
        self.flush()
        for attr_name in ("masks", "state"):
            memmap_obj = getattr(self, attr_name, None)
            if memmap_obj is None:
                continue
            mmap_obj = getattr(memmap_obj, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
            setattr(self, attr_name, None)


def composite(
    frame: np.ndarray,
    mask: np.ndarray | None,
    bg_frame: np.ndarray,
) -> np.ndarray:
    if mask is None or not mask.any():
        return bg_frame.copy()
    out = frame.copy()
    out[~mask] = bg_frame[~mask]
    return out


def merge_audio(tmp_file: Path, source_video: str, output_file: Path):
    try:
        (
            ffmpeg.output(
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
        warnings.warn(f"Audio passthrough failed - saving without audio. ({stderr})")
        tmp_file.rename(output_file)


def propagate_masks(
    backend: Sam3AppBackend,
    prompt_frame_idx: int,
    total_frames: int,
    mask_store: PackedMaskStore,
):
    stored_count = 0

    def handle_stream(stream, label: str):
        nonlocal stored_count
        print(label)
        for frame_idx, mask in stream:
            if mask is not None:
                mask_store.put(frame_idx, mask)
                stored_count += 1
            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx} / {total_frames}")

    handle_stream(
        backend.propagate_masks(prompt_frame_idx, total_frames),
        f"Propagating from frame {prompt_frame_idx}...",
    )

    mask_store.flush()
    return stored_count


def build_background_frame(height: int, width: int, bg_value: int) -> np.ndarray:
    return np.full((height, width, 3), bg_value, dtype=np.uint8)


def compute_mask_stats(mask: np.ndarray) -> dict[str, float]:
    coords = np.argwhere(mask)
    area = float(coords.shape[0])
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    centroid_y = float(coords[:, 0].mean())
    centroid_x = float(coords[:, 1].mean())
    return {
        "area": area,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "bbox_w": float(x_max - x_min + 1),
        "bbox_h": float(y_max - y_min + 1),
    }


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum(dtype=np.int64)
    union = np.logical_or(mask_a, mask_b).sum(dtype=np.int64)
    if union == 0:
        return 0.0
    return float(intersection / union)


def is_low_confidence_mask(
    mask: np.ndarray,
    current_stats: dict[str, float],
    previous_mask: np.ndarray | None,
    previous_stats: dict[str, float] | None,
) -> bool:
    if previous_mask is None or previous_stats is None:
        return False

    prev_area = max(previous_stats["area"], 1.0)
    area_ratio = current_stats["area"] / prev_area
    centroid_dx = (current_stats["centroid_x"] - previous_stats["centroid_x"]) / mask.shape[1]
    centroid_dy = (current_stats["centroid_y"] - previous_stats["centroid_y"]) / mask.shape[0]
    centroid_jump = math.hypot(centroid_dx, centroid_dy)
    iou = compute_mask_iou(mask, previous_mask)

    area_implausible = area_ratio < 0.15 or area_ratio > 6.0
    motion_implausible = centroid_jump > 0.35 and iou < 0.01
    return area_implausible or motion_implausible


def interpolate_masks(mask_a: np.ndarray, mask_b: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return mask_a.copy()
    if alpha >= 1.0:
        return mask_b.copy()
    blended = (1.0 - alpha) * mask_a.astype(np.float32) + alpha * mask_b.astype(np.float32)
    return blended >= 0.5


def analyze_mask_sequence(
    mask_store: PackedMaskStore,
) -> tuple[list[str], list[str], list[np.ndarray | None]]:
    quality_states = [FALLBACK_STATE_MISSING] * mask_store.total_frames
    render_states = [FALLBACK_STATE_MISSING] * mask_store.total_frames
    resolved_masks: list[np.ndarray | None] = [None] * mask_store.total_frames
    raw_states: list[int] = [mask_store.get_state(frame_idx) for frame_idx in range(mask_store.total_frames)]

    last_good_mask = None
    last_good_stats = None
    for frame_idx, raw_state in enumerate(raw_states):
        if raw_state == MASK_STATE_MISSING:
            continue
        mask = mask_store.get(frame_idx)
        if mask is None:
            continue
        if raw_state == MASK_STATE_EMPTY:
            quality_states[frame_idx] = FALLBACK_STATE_EMPTY
            continue

        current_stats = compute_mask_stats(mask)
        if is_low_confidence_mask(mask, current_stats, last_good_mask, last_good_stats):
            quality_states[frame_idx] = FALLBACK_STATE_LOW_CONFIDENCE
            continue

        quality_states[frame_idx] = FALLBACK_STATE_VALID
        render_states[frame_idx] = FALLBACK_STATE_VALID
        resolved_masks[frame_idx] = mask
        last_good_mask = mask
        last_good_stats = current_stats

    reliable_indices = [
        frame_idx for frame_idx, state in enumerate(quality_states) if state == FALLBACK_STATE_VALID
    ]

    max_carry_missing = 5
    max_carry_empty = 3
    max_interp_gap = 3

    for frame_idx, state in enumerate(quality_states):
        if state == FALLBACK_STATE_VALID:
            continue

        prev_idx = next(
            (idx for idx in range(frame_idx - 1, -1, -1) if quality_states[idx] == FALLBACK_STATE_VALID),
            None,
        )
        next_idx = next(
            (idx for idx in range(frame_idx + 1, mask_store.total_frames) if quality_states[idx] == FALLBACK_STATE_VALID),
            None,
        )

        gap_prev = None if prev_idx is None else frame_idx - prev_idx
        gap_next = None if next_idx is None else next_idx - frame_idx

        can_interpolate = (
            prev_idx is not None
            and next_idx is not None
            and (next_idx - prev_idx - 1) <= max_interp_gap
            and state in {FALLBACK_STATE_EMPTY, FALLBACK_STATE_LOW_CONFIDENCE, FALLBACK_STATE_MISSING}
        )
        if can_interpolate:
            alpha = (frame_idx - prev_idx) / (next_idx - prev_idx)
            resolved_masks[frame_idx] = interpolate_masks(
                resolved_masks[prev_idx],
                resolved_masks[next_idx],
                alpha,
            )
            render_states[frame_idx] = FALLBACK_STATE_INTERPOLATED
            continue

        carry_limit = max_carry_empty if state in {FALLBACK_STATE_EMPTY, FALLBACK_STATE_LOW_CONFIDENCE} else max_carry_missing
        if prev_idx is not None and gap_prev is not None and gap_prev <= carry_limit:
            resolved_masks[frame_idx] = resolved_masks[prev_idx].copy()
            render_states[frame_idx] = FALLBACK_STATE_CARRIED
            continue

        render_states[frame_idx] = FALLBACK_STATE_LOST

    if not reliable_indices:
        quality_states = [FALLBACK_STATE_MISSING] * mask_store.total_frames
        render_states = [FALLBACK_STATE_LOST] * mask_store.total_frames

    return quality_states, render_states, resolved_masks


def build_ffmpeg_writer(
    tmp_file: Path,
    width: int,
    height: int,
    native_fps: float,
    output_fps: float | None,
):
    stream = ffmpeg.input(
        "pipe:",
        format="rawvideo",
        pix_fmt="bgr24",
        s=f"{width}x{height}",
        r=native_fps,
    )
    if output_fps is not None and not math.isclose(output_fps, native_fps):
        stream = stream.filter("fps", fps=output_fps, round="near")

    return (
        ffmpeg.output(stream, str(tmp_file), pix_fmt="yuv420p", vcodec="libx264", crf=18)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )


def main():
    args = parse_args()

    if args.prompt is not None:
        print("Warning: --prompt is not yet implemented. Using visual prompt mode.")

    if not os.path.exists(args.video_path):
        sys.exit(f"Error: Video file not found: '{args.video_path}'")

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video_path).stem
    output_file = output_dir / f"{stem}_bg_removed.mp4"
    tmp_file = output_dir / f"{stem}.noaudio.tmp.mp4"

    bg_value = 255 if args.bg_color == "white" else 0

    cap, native_fps, width, height, total_frames = open_video(args.video_path)
    if total_frames <= 0:
        cap.release()
        sys.exit("Error: Could not determine the total number of frames.")
    if native_fps <= 0:
        cap.release()
        sys.exit("Error: Could not determine a valid FPS from the input video.")
    if args.fps is not None and args.fps <= 0:
        cap.release()
        sys.exit("Error: --fps must be greater than 0.")

    prompt_frame_idx = min(round(native_fps), total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, prompt_frame_idx)
    ret, prompt_frame = cap.read()
    cap.release()
    if not ret:
        sys.exit(f"Error: Could not read prompt frame at index {prompt_frame_idx}.")

    prompt_input = collect_prompt(prompt_frame, width, height)
    if prompt_input is None:
        print("Cancelled.")
        sys.exit(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device != "cuda":
        sys.exit("Error: This script currently requires CUDA-enabled inference.")

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    weights_dir = Path(__file__).parent / "weights"
    checkpoint_path, model_version = resolve_checkpoint(args, weights_dir)
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Model version: {model_version}")

    backend = None
    backend, backend_name = load_backend(checkpoint_path, model_version)
    print(f"Backend: {backend_name}")
    if getattr(backend, "video_loader_type", None) is not None:
        print(f"Video loader: {backend.video_loader_type}")

    try:
        backend.start_session(args.video_path)
        backend.add_visual_prompt(prompt_input, prompt_frame_idx, width, height)

        with tempfile.TemporaryDirectory(prefix=f"{stem}_masks_", dir=output_dir) as tmp_dir:
            mask_store = PackedMaskStore(Path(tmp_dir), total_frames, height, width)
            try:
                stored_count = propagate_masks(
                    backend,
                    prompt_frame_idx=prompt_frame_idx,
                    total_frames=total_frames,
                    mask_store=mask_store,
                )
                print(f"Collected masks for {stored_count} frame passes across {total_frames} frames.")
                quality_states, render_states, resolved_masks = analyze_mask_sequence(mask_store)
            finally:
                mask_store.close()
                gc.collect()

            ffmpeg_proc = build_ffmpeg_writer(
                tmp_file=tmp_file,
                width=width,
                height=height,
                native_fps=native_fps,
                output_fps=args.fps,
            )

            cap = cv2.VideoCapture(args.video_path)
            bg_frame = build_background_frame(height, width, bg_value)
            written = 0
            print("Compositing and writing output...")
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                mask = resolved_masks[frame_idx]
                out_frame = composite(frame, mask, bg_frame)
                ffmpeg_proc.stdin.write(out_frame.tobytes())
                written += 1

            cap.release()
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()

        quality_summary = {
            state: sum(1 for current_state in quality_states if current_state == state)
            for state in [
                FALLBACK_STATE_VALID,
                FALLBACK_STATE_LOW_CONFIDENCE,
                FALLBACK_STATE_EMPTY,
                FALLBACK_STATE_MISSING,
            ]
        }
        render_summary = {
            state: sum(1 for current_state in render_states if current_state == state)
            for state in [
                FALLBACK_STATE_VALID,
                FALLBACK_STATE_CARRIED,
                FALLBACK_STATE_INTERPOLATED,
                FALLBACK_STATE_LOST,
            ]
        }
        print("Mask quality summary:")
        for state, count in quality_summary.items():
            print(f"  {state}: {count}")
        print("Render decision summary:")
        for state, count in render_summary.items():
            print(f"  {state}: {count}")

        print(f"Composited {written} frames. Merging audio...")
        merge_audio(tmp_file, args.video_path, output_file)
        print(f"Done. Output: {output_file}")
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    main()
