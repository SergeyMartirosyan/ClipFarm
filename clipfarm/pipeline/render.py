"""ffmpeg: cut + 9:16 layout + burned captions, plus a small preview for Telegram."""

from __future__ import annotations

from pathlib import Path

from ..media import run
from .layout import CAM_H, GAME_H, OUT_H, OUT_W, Layout


def _crop(b) -> str:
    return f"crop={b.w}:{b.h}:{b.x}:{b.y}"


def _ass(path: Path) -> str:
    # The path is ours (no quotes/colons), but escape defensively for the filter parser.
    p = str(path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    return f"ass='{p}'"


def video_filter(layout: Layout, ass_path: Path | None, base_ass: Path | None | bool = False) -> str:
    """filter_complex graph from input [0:v] to output [v]. With base_ass (a path, or None for no
    captions) a second output [vb] is made too: the same picture without the hook text, kept so the
    hook can be swapped later without going back to the source video."""
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
    main = "stack"
    if base_ass is not False:
        graph += ";[stack]split=2[main][stackb]"
        graph += f";[stackb]{_ass(base_ass) if base_ass is not None else 'null'}[vb]"
        main = "main"
    graph += f";[{main}]{_ass(ass_path) if ass_path is not None else 'null'}[v]"
    return graph


def caption_margin(layout: Layout) -> int:
    """Spoken captions: lower third (bottom margin in px), clear of the hook text."""
    return 470 if layout.kind == "facecam" else 560


def hook_position(layout) -> tuple[int, int, int]:
    """(ASS alignment, x, y) for the hook text box (layout: a Layout or its kind).
    facecam: centred on the seam between webcam and game; blur: just above the video;
    full-frame layouts: near the top."""
    kind = getattr(layout, "kind", layout)
    if kind == "facecam":
        return 5, OUT_W // 2, CAM_H
    if kind == "blur":
        return 2, OUT_W // 2, 620
    return 8, OUT_W // 2, 230


AUDIO = ["-c:a", "aac", "-b:a", "160k", "-ar", "48000"]


def render_clip(
    src: Path, start: float, end: float, layout: Layout, ass_path: Path | None, out: Path, has_audio: bool = True,
    base_out: Path | None = None, base_ass: Path | None = None,
) -> Path:
    """The final clip. With base_out, the same clip without the hook text is written too, in the
    same pass (ultrafast + high quality: it is only a master for re-adding a different hook)."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
        "-filter_complex", video_filter(layout, ass_path, base_ass if base_out else False),
        "-map", "[v]",
    ]
    if has_audio:
        cmd += ["-map", "0:a:0", *AUDIO]
    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", "30",
        "-movflags", "+faststart",
        str(out),
    ]
    if base_out:
        cmd += ["-map", "[vb]"]
        if has_audio:
            cmd += ["-map", "0:a:0", *AUDIO]
        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-pix_fmt", "yuv420p", "-r", "30",
                str(base_out)]
    run(cmd)
    return out


def rerender_hook(base: Path, hook: str, layout_kind: str, duration: float, out: Path) -> Path:
    """A clip with a different hook text: the hook-free master plus the new hook (captions are
    already in the master). Written next to `out` first, then moved over it."""
    from .captions import write_ass

    ass = out.with_name(out.stem + ".hook.ass")
    write_ass([], ass, hook=hook, hook_pos=hook_position(layout_kind), duration=duration)
    tmp = out.with_name(out.stem + ".new.mp4")
    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(base), "-vf", _ass(ass),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-movflags", "+faststart", str(tmp),
    ])
    tmp.replace(out)
    ass.unlink(missing_ok=True)
    return out


def render_preview(final: Path, out: Path) -> Path:
    """~5-10 MB 720p copy: Telegram bots can only upload up to 50 MB."""
    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(final),
        "-vf", "scale=720:1280", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(out),
    ])
    return out
