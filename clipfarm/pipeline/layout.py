"""Decide how a horizontal clip becomes 9:16.

Layouts:
  facecam  - small webcam overlay found: webcam crop on top (40%), gameplay below (60%)
  face     - a face fills the frame (podcast, vlog): 9:16 crop centred on it
  blur     - no usable face: whole frame centred over a blurred copy of itself
  vertical - the source is already vertical: just scale
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

OUT_W, OUT_H = 1080, 1920
CAM_H = 768                 # top panel for the facecam layout (40%)
GAME_H = OUT_H - CAM_H      # bottom panel (60%)

# Face width (fraction of frame width) below which we treat it as a webcam overlay.
FACECAM_MAX_FACE_W = 0.15


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Layout:
    kind: str                  # facecam | face | blur | vertical
    cam: Box | None = None     # facecam: webcam crop
    main: Box | None = None    # facecam: gameplay crop; face: the 9:16 crop


def _even(v: float) -> int:
    return max(2, int(round(v / 2.0)) * 2)


def _fit_box(cx: float, cy: float, w: float, h: float, fw: int, fh: int) -> Box:
    """Box of size w x h centred on (cx, cy), shifted/shrunk to stay inside the frame."""
    w, h = min(w, fw), min(h, fh)
    x = min(max(cx - w / 2, 0), fw - w)
    y = min(max(cy - h / 2, 0), fh - h)
    return Box(_even(x) if x > 1 else 0, _even(y) if y > 1 else 0, _even(w), _even(h))


def facecam_layout(face: tuple[float, float, float, float], fw: int, fh: int) -> Layout:
    """face = (x, y, w, h) in pixels of the median detected webcam face."""
    fx, fy, fwid, fhgt = face
    cam_aspect = OUT_W / CAM_H  # 1.406
    cam_w = fwid * 3.6
    cam_h = cam_w / cam_aspect
    if cam_h > fh:
        cam_h = fh
        cam_w = cam_h * cam_aspect
    cam = _fit_box(fx + fwid / 2, fy + fhgt / 2, cam_w, cam_h, fw, fh)

    game_aspect = OUT_W / GAME_H  # 0.9375
    game_w = min(fw, fh * game_aspect)
    game = _fit_box(fw / 2, fh / 2, game_w, game_w / game_aspect, fw, fh)
    return Layout("facecam", cam=cam, main=game)


def face_layout(face_cx: float, fw: int, fh: int) -> Layout:
    w = fh * OUT_W / OUT_H
    return Layout("face", main=_fit_box(face_cx, fh / 2, w, fh, fw, fh))


def preset_layout(preset: dict, fw: int, fh: int) -> Layout:
    def box(v) -> Box:
        x, y, w, h = v
        return Box(_even(x * fw), _even(y * fh), _even(w * fw), _even(h * fh))

    return Layout("facecam", cam=box(preset["cam"]), main=box(preset["game"]))


def load_presets(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {k.lower(): v for k, v in data.items() if not k.startswith("_")}


def decide(faces: list[list[tuple[float, float, float, float]]], fw: int, fh: int) -> Layout:
    """faces: per sampled frame, the detected face boxes (pixels)."""
    if fh > fw:
        return Layout("vertical")
    frames_with_face = [f for f in faces if f]
    if not faces or len(frames_with_face) < 0.4 * len(faces):
        return Layout("blur")
    # Most prominent face per frame, then the median position over time.
    biggest = [max(f, key=lambda b: b[2] * b[3]) for f in frames_with_face]
    med = tuple(float(np.median([b[i] for b in biggest])) for i in range(4))
    # A face that jumps around isn't a fixed webcam; a single median crop would miss it.
    spread = float(np.median([abs((b[0] + b[2] / 2) - (med[0] + med[2] / 2)) for b in biggest]))
    if med[2] / fw < FACECAM_MAX_FACE_W:
        if spread > fw * 0.1:
            return Layout("blur")
        return facecam_layout(med, fw, fh)
    return face_layout(med[0] + med[2] / 2, fw, fh)


def detect_faces(video: Path, start: float, end: float, model_path: Path, samples: int = 12):
    """Face boxes on `samples` frames spread across [start, end]."""
    import cv2

    if not Path(model_path).exists():
        log.warning("Face model missing at %s; using blur layout", model_path)
        return None
    cap = cv2.VideoCapture(str(video))
    detector = None
    results: list[list[tuple[float, float, float, float]]] = []
    try:
        for i in range(samples):
            t = start + (end - start) * (i + 0.5) / samples
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            scale = 640.0 / w if w > 640 else 1.0
            small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale != 1.0 else frame
            sh, sw = small.shape[:2]
            if detector is None:
                detector = cv2.FaceDetectorYN.create(str(model_path), "", (sw, sh), 0.7)
            detector.setInputSize((sw, sh))
            _, dets = detector.detect(small)
            boxes = []
            if dets is not None:
                for d in dets:
                    x, y, bw, bh = (float(v) / scale for v in d[:4])
                    boxes.append((x, y, bw, bh))
            results.append(boxes)
    finally:
        cap.release()
    return results


def choose_layout(
    video: Path, start: float, end: float, fw: int, fh: int, model_path: Path,
    uploader: str = "", presets: dict | None = None,
) -> Layout:
    if fh > fw:
        return Layout("vertical")
    preset = (presets or {}).get((uploader or "").lower())
    if preset:
        return preset_layout(preset, fw, fh)
    faces = detect_faces(video, start, end, model_path)
    if faces is None:
        return Layout("blur")
    return decide(faces, fw, fh)
