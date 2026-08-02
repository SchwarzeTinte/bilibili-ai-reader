from __future__ import annotations

import shutil
import subprocess
from math import ceil
from pathlib import Path
from typing import Callable

from .llm import LLMSettings, describe_images
from .models import Segment, Transcript
from .storage import video_directory
from .text import format_timestamp


def frame_sampling_plan(duration: float) -> tuple[int, float]:
    """Choose denser sampling for short videos while bounding cost on long ones."""
    effective_duration = max(1.0, float(duration or 0))
    if effective_duration <= 10 * 60:
        preferred_interval = 10.0
    elif effective_duration <= 30 * 60:
        preferred_interval = 15.0
    else:
        preferred_interval = 30.0
    frame_count = max(12, min(180, ceil(effective_duration / preferred_interval)))
    interval = max(1.0, effective_duration / frame_count)
    return frame_count, interval


def vision_batch_size(settings: LLMSettings) -> int:
    """Use single-frame requests for local models to avoid cross-frame confusion."""
    is_local = settings.provider == "Ollama" or (
        settings.provider == "OpenAI 兼容（自定义）"
        and any(
            host in settings.base_url.lower()
            for host in ("localhost", "127.0.0.1", "0.0.0.0")
        )
    )
    return 1 if is_local else 3


def extract_frames(
    video_path: Path,
    *,
    video_id: str,
    duration: float,
    max_frames: int | None = None,
) -> list[tuple[float, Path]]:
    """Sample a bounded number of frames so long videos do not explode API cost."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("没有找到 FFmpeg，无法从视频中提取画面。请先安装 FFmpeg。")
    frame_dir = video_directory(video_id) / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("frame-*.jpg"):
        old_frame.unlink(missing_ok=True)

    effective_duration = max(1.0, float(duration or 0))
    planned_frames, interval = frame_sampling_plan(effective_duration)
    max_frames = max_frames or planned_frames
    interval = max(1.0, effective_duration / max_frames)
    output_pattern = str(frame_dir / "frame-%04d.jpg")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval},scale=min(1440\\,iw):-2",
        "-frames:v",
        str(max_frames),
        "-q:v",
        "3",
        output_pattern,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-800:] or "FFmpeg 未返回具体原因"
        raise RuntimeError(f"视频画面提取失败：{detail}")
    paths = sorted(frame_dir.glob("frame-*.jpg"))
    if not paths:
        raise RuntimeError("视频中没有提取到可分析的画面。")
    return [(min(index * interval, effective_duration), path) for index, path in enumerate(paths)]


def analyze_video_frames(
    video_path: Path,
    *,
    video_id: str,
    title: str,
    duration: float,
    settings: LLMSettings,
    progress: Callable[[int, str], None] | None = None,
) -> Transcript:
    planned_frames, interval = frame_sampling_plan(duration)
    if progress:
        progress(
            10,
            f"正在按时间抽取约 {planned_frames} 张代表性画面（平均每 {interval:.0f} 秒一张）……",
        )
    frames = extract_frames(
        video_path,
        video_id=video_id,
        duration=duration,
        max_frames=planned_frames,
    )
    batch_size = vision_batch_size(settings)
    segments: list[Segment] = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        labels = [(f"[{format_timestamp(timestamp)}]", path) for timestamp, path in batch]
        if progress:
            percent = 20 + int(70 * start / max(1, len(frames)))
            progress(percent, f"正在分析画面 {start + 1}–{start + len(batch)} / {len(frames)}……")
        try:
            description = describe_images(settings, labels)
        except Exception as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            raise RuntimeError(
                "当前模型无法完成图片分析。请在设置中选择明确支持视觉/图片输入的模型后重试。"
                f"\n\n原始错误：{detail}"
            ) from exc
        segment_start = batch[0][0]
        segment_end = batch[-1][0] + max(1.0, duration / max(1, len(frames)))
        segments.append(Segment(segment_start, min(segment_end, duration or segment_end), description))
    if progress:
        progress(100, "画面内容已整理为可总结、可问答的时间线。")
    return Transcript(
        video_id=video_id,
        title=title,
        source=f"画面分析 · {settings.provider} {settings.model}",
        language="visual",
        segments=segments,
    )
