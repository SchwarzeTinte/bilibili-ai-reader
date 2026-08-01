from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .models import Segment, Transcript


ProgressCallback = Callable[[int, str], None]


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
    model = WhisperModel(model_size, device="auto", compute_type="default")
    if progress:
        progress(20, "模型已加载，正在识别语音……")
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
