from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable

from .models import Segment, Transcript


ProgressCallback = Callable[[int, str], None]


def _is_cuda_runtime_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    markers = (
        "cuda",
        "cublas",
        "cudnn",
        "nvcuda",
        "cannot be loaded",
        "could not load library",
    )
    return any(marker in message for marker in markers)


def _run_whisper(
    model_class: Any,
    audio_path: Path,
    model_size: str,
    language: str | None,
    device: str,
    compute_type: str,
    progress: ProgressCallback | None,
) -> tuple[list[Segment], Any]:
    model = model_class(model_size, device=device, compute_type=compute_type)
    if progress:
        mode = "GPU" if device != "cpu" else "CPU int8"
        progress(20, f"模型已加载（{mode}），正在识别语音……")
    generated, info = model.transcribe(
        str(audio_path),
        language=language or None,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=True,
    )
    duration = max(float(getattr(info, "duration", 0) or 0), 1.0)
    segments: list[Segment] = []
    for item in generated:
        text = item.text.strip()
        if text:
            segments.append(Segment(float(item.start), float(item.end), text))
        if progress:
            percent = min(98, 20 + int(78 * float(item.end) / duration))
            progress(percent, f"正在转写 {int(item.end)} / {int(duration)} 秒……")
    return segments, info


def transcribe_audio(
    audio_path: Path,
    video_id: str,
    title: str,
    model_size: str = "small",
    language: str | None = "zh",
    progress: ProgressCallback | None = None,
) -> Transcript:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("尚未安装 faster-whisper，请先运行安装脚本。") from exc

    if progress:
        progress(5, f"正在加载 Whisper {model_size} 模型，首次运行会自动下载模型……")
    try:
        segments, info = _run_whisper(
            WhisperModel,
            audio_path,
            model_size,
            language,
            device="auto",
            compute_type="default",
            progress=progress,
        )
    except (RuntimeError, OSError) as exc:
        if not _is_cuda_runtime_error(exc):
            raise
        gc.collect()
        if progress:
            progress(10, "检测到 CUDA 运行库不完整，已自动切换到 CPU int8 模式……")
        segments, info = _run_whisper(
            WhisperModel,
            audio_path,
            model_size,
            language,
            device="cpu",
            compute_type="int8",
            progress=progress,
        )
    if not segments:
        raise RuntimeError("Whisper 没有从音频中识别出文字。")
    if progress:
        progress(100, "语音转写完成。")
    detected_language = str(getattr(info, "language", None) or language or "unknown")
    return Transcript(
        video_id=video_id,
        title=title,
        source=f"Whisper {model_size}",
        language=detected_language,
        segments=segments,
    )
