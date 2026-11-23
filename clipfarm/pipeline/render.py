"""ffmpeg: cut + 9:16 layout + burned captions, plus a small preview for Telegram."""

from __future__ import annotations

from pathlib import Path

from ..media import run
from .layout import CAM_H, GAME_H, OUT_H, OUT_W, Layout


def _crop(b) -> str:
    return f"crop={b.w}:{b.h}:{b.x}:{b.y}"


def video_filter(layout: Layout, ass_path: Path | None) -> str:
    """filter_complex graph from input [0:v] to output [v]."""
    if layout.kind == "facecam":
        graph = (
            f"[0:v]split=2[a][b];"
            f"[a]{_crop(layout.cam)},scale={OUT_W}:{CAM_H},setsar=1[top];"
            f"[b]{_crop(layout.main)},scale={OUT_W}:{GAME_H},setsar=1[bot];"
            f"[top][bot]vstack=inputs=2[stack]"
        )
    elif layout.kind == "face":
        graph = f"[0:v]{_crop(layout.main)},scale={OUT_W}:{OUT_H},setsar=1[stack]"
    elif layout.kind == "vertical":
        graph = (
            f"[0:v]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
            f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,setsar=1[stack]"
        )
    else:  # blur
        graph = (
            f"[0:v]split=2[a][b];"
            f"[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},"
            f"boxblur=20:5,eq=brightness=-0.08[bg];"
            f"[b]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[stack]"
        )
    if ass_path is not None:
        # The path is ours (no quotes/colons), but escape defensively for the filter parser.
        p = str(ass_path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        graph += f";[stack]ass='{p}'[v]"
    else:
        graph += ";[stack]null[v]"
    return graph


def caption_margin(layout: Layout) -> int:
    """Captions sit just inside the gameplay panel for facecam, lower-middle otherwise."""
    return OUT_H - CAM_H - 260 if layout.kind == "facecam" else 560


def render_clip(
    src: Path, start: float, end: float, layout: Layout, ass_path: Path | None, out: Path, has_audio: bool = True
) -> Path:
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
        "-filter_complex", video_filter(layout, ass_path),
        "-map", "[v]",
    ]
    if has_audio:
        cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "160k", "-ar", "48000"]
    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", "30",
        "-movflags", "+faststart",
        str(out),
    ]
    run(cmd)
    return out


def render_preview(final: Path, out: Path) -> Path:
    """~5-10 MB 720p copy: Telegram bots can only upload up to 50 MB."""
    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(final),
        "-vf", "scale=720:1280", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(out),
    ])
    return out
