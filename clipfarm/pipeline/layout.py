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
        x, y, w, h = (min(max(float(n), 0.0), 1.0) for n in v)
        bw, bh = _even(max(w, 0.02) * fw), _even(max(h, 0.02) * fh)
        bw, bh = min(bw, fw - fw % 2), min(bh, fh - fh % 2)
        bx = min(_even(x * fw) if x > 0 else 0, fw - bw)
        by = min(_even(y * fh) if y > 0 else 0, fh - bh)
        return Box(max(bx, 0), max(by, 0), bw, bh)

    return Layout("facecam", cam=box(preset["cam"]), main=box(preset["game"]))


def load_presets(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {k.lower(): v for k, v in data.items() if not k.startswith("_")}


FaceBox = tuple[float, float, float, float]  # x, y, w, h in pixels

# A webcam is a face at the same spot in at least this share of frames sampled across the stream.
CAM_MIN_SHARE = 0.25
# In a clip, the webcam counts as visible if its face is found in this share of the clip's frames.
CAM_PRESENT_SHARE = 0.25


def _center(b: FaceBox) -> tuple[float, float]:
    return b[0] + b[2] / 2, b[1] + b[3] / 2


def webcam_from_faces(samples: list[list[FaceBox]], fw: int, fh: int) -> FaceBox | None:
    """The streamer's webcam: a small face that sits at the same spot in a large share of frames
    sampled across the whole stream. Faces inside the game (cutscenes, characters) move around and
    come and go, so they don't qualify. Returns the median face box, or None."""
    small = [(i, b) for i, faces in enumerate(samples) for b in faces if b[2] / fw < FACECAM_MAX_FACE_W]
    if not small:
        return None
    tol_x, tol_y = 0.04 * fw, 0.06 * fh

    def near(a: FaceBox, b: FaceBox) -> bool:
        (ax, ay), (bx, by) = _center(a), _center(b)
        return abs(ax - bx) <= tol_x and abs(ay - by) <= tol_y

    best_frames: set[int] = set()
    best_anchor: FaceBox | None = None
    for _, anchor in small:
        frames = {i for i, b in small if near(anchor, b)}
        if len(frames) > len(best_frames):
            best_frames, best_anchor = frames, anchor
    if best_anchor is None or len(best_frames) < max(4, CAM_MIN_SHARE * len(samples)):
        return None
    members = [b for _, b in small if near(best_anchor, b)]
    return tuple(float(np.median([b[k] for b in members])) for k in range(4))  # type: ignore[return-value]


def webcam_visible(faces: list[list[FaceBox]], cam: FaceBox, fw: int, fh: int) -> bool:
    """Is the stream's webcam showing in this clip? (Streamers often hide it during cutscenes.)"""
    if not faces:
        return False
    cx, cy = _center(cam)
    tol_x, tol_y = max(cam[2], 0.04 * fw), max(cam[3], 0.06 * fh)  # the head moves inside the cam
    hits = sum(
        1 for frame in faces
        if any(abs(_center(b)[0] - cx) <= tol_x and abs(_center(b)[1] - cy) <= tol_y for b in frame)
    )
    return hits >= max(2, CAM_PRESENT_SHARE * len(faces))


def decide(faces: list[list[FaceBox]], fw: int, fh: int, allow_facecam: bool = True) -> Layout:
    """faces: per sampled frame, the detected face boxes (pixels). With allow_facecam=False (the
    stream has no fixed webcam, or it isn't showing), small faces are game characters, not a cam."""
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
        if spread > fw * 0.1 or not allow_facecam:
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


def find_stream_webcam(video: Path, duration: float, fw: int, fh: int, model_path: Path,
                       samples: int = 80) -> FaceBox | None:
    """Look for a fixed webcam across the whole stream (one pass, ~30-60 s on a long VOD)."""
    if fh > fw or duration <= 0:
        return None
    faces = detect_faces(video, 0, duration, model_path, samples=samples)
    return webcam_from_faces(faces, fw, fh) if faces else None


def choose_layout(
    video: Path, start: float, end: float, fw: int, fh: int, model_path: Path,
    uploader: str = "", presets: dict | None = None,
    stream_cam: FaceBox | None = None, cam_searched: bool = False,
) -> Layout:
    """stream_cam/cam_searched come from find_stream_webcam. When the stream was searched, only
    its fixed webcam can become the top panel, and only in clips where it is actually showing."""
    if fh > fw:
        return Layout("vertical")
    preset = (presets or {}).get((uploader or "").lower())
    if preset:
        return preset_layout(preset, fw, fh)
    faces = detect_faces(video, start, end, model_path)
    if faces is None:
        return Layout("blur")
    if stream_cam is not None:
        if webcam_visible(faces, stream_cam, fw, fh):
            return facecam_layout(stream_cam, fw, fh)
        return decide(faces, fw, fh, allow_facecam=False)  # cam hidden in this clip: show the game
    return decide(faces, fw, fh, allow_facecam=not cam_searched)
