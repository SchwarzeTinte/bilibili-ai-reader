from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .extractor import download_audio, download_subtitle, download_video
from .llm import LLMSettings, vision_support_status
from .models import Segment, Transcript
from .transcriber import transcribe_audio
from .visual import analyze_video_frames


@dataclass(slots=True)
class TextQuality:
    effective_units: int
    units_per_minute: float
    timeline_coverage: float
    sufficient: bool
    reason: str


@dataclass(slots=True)
class SmartReadResult:
    transcript: Transcript
    message: str
    warnings: list[str] = field(default_factory=list)
    used_visual: bool = False


def transcript_text_quality(
    transcript: Transcript,
    duration: float,
    *,
    units_per_minute_threshold: int = 30,
    timeline_threshold: float = 0.35,
) -> TextQuality:
    text = " ".join(segment.text for segment in transcript.segments)
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    latin_words = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", text))
    effective_units = cjk + latin_words * 2
    effective_duration = max(1.0, float(duration or 0))
    minutes = max(1.0, effective_duration / 60)
    units_per_minute = effective_units / minutes
    bucket_count = 10
    covered_buckets: set[int] = set()
    for segment in transcript.segments:
        start_bucket = min(
            bucket_count - 1,
            max(0, int(max(0.0, segment.start) / effective_duration * bucket_count)),
        )
        end_bucket = min(
            bucket_count - 1,
            max(0, int(max(segment.start, segment.end) / effective_duration * bucket_count)),
        )
        covered_buckets.update(range(start_bucket, end_bucket + 1))
    timeline_coverage = len(covered_buckets) / bucket_count

    minimum_units = max(80, int(minutes * units_per_minute_threshold))
    enough_text = effective_units >= minimum_units
    enough_timeline = effective_duration <= 180 or timeline_coverage >= timeline_threshold
    sufficient = enough_text and enough_timeline
    reasons: list[str] = []
    if not enough_text:
        reasons.append(
            f"有效文字约 {effective_units} 单位（{units_per_minute:.0f}/分钟），低于建议值 {minimum_units}"
        )
    if not enough_timeline:
        reasons.append(f"文字时间线仅覆盖约 {timeline_coverage:.0%}")
    return TextQuality(
        effective_units=effective_units,
        units_per_minute=units_per_minute,
        timeline_coverage=timeline_coverage,
        sufficient=sufficient,
        reason="；".join(reasons) if reasons else "文字数量和时间线覆盖充足",
    )


def merge_text_and_visual(text: Transcript | None, visual: Transcript) -> Transcript:
    if text is None:
        return visual
    segments = [*text.segments, *visual.segments]
    segments.sort(key=lambda segment: (segment.start, segment.end))
    return Transcript(
        video_id=text.video_id,
        title=text.title,
        source=f"{text.source} + {visual.source}",
        language=text.language,
        segments=segments,
    )


def smart_read_video(
    part: dict[str, object],
    *,
    auth_source: str | None,
    whisper_model: str,
    whisper_language: str | None,
    whisper_device: str,
    settings: LLMSettings,
    progress: Callable[[int, str], None],
    visual_fallback_sensitivity: str = "标准（推荐）",
) -> SmartReadResult:
    duration = float(part.get("duration") or 0)
    warnings: list[str] = []
    candidates: list[tuple[Transcript, TextQuality]] = []
    tracks = list(part.get("tracks") or [])
    sensitivity_thresholds = {
        "节省费用": (18, 0.25),
        "标准（推荐）": (30, 0.35),
        "严格完整": (50, 0.50),
    }
    density_threshold, timeline_threshold = sensitivity_thresholds.get(
        visual_fallback_sensitivity,
        sensitivity_thresholds["标准（推荐）"],
    )

    def quality_of(transcript: Transcript) -> TextQuality:
        return transcript_text_quality(
            transcript,
            duration,
            units_per_minute_threshold=density_threshold,
            timeline_threshold=timeline_threshold,
        )

    if tracks:
        try:
            progress(5, "检测到字幕，正在优先提取并评估文字完整度……")
            subtitle = download_subtitle(part, tracks[0])
            quality = quality_of(subtitle)
            candidates.append((subtitle, quality))
            if quality.sufficient:
                return SmartReadResult(subtitle, "已采用完整度充足的现有字幕。")
            warnings.append(f"现有字幕内容偏少：{quality.reason}。")
        except Exception as exc:
            warnings.append(f"字幕不可用：{exc}")

    if part.get("has_audio") is not False:
        try:
            progress(15, "现有文字不足，正在下载音频并使用 Whisper 补充……")
            audio_path = download_audio(part, auth_source)

            def audio_progress(percent: int, message: str) -> None:
                progress(max(15, min(70, 15 + percent * 55 // 100)), message)

            audio_transcript = transcribe_audio(
                audio_path,
                video_id=str(part["id"]),
                title=str(part["title"]),
                model_size=whisper_model,
                language=whisper_language,
                device_mode=whisper_device,
                progress=audio_progress,
            )
            audio_quality = quality_of(audio_transcript)
            candidates.append((audio_transcript, audio_quality))
            if audio_quality.sufficient:
                return SmartReadResult(
                    audio_transcript,
                    "现有字幕不足，已自动采用完整度更高的 Whisper 转写。",
                    warnings,
                )
            warnings.append(f"音频转写的有效文字仍偏少：{audio_quality.reason}。")
        except Exception as exc:
            warnings.append(f"音频不可用：{exc}")

    best_text = max(candidates, key=lambda item: item[1].effective_units)[0] if candidates else None
    support, support_message = vision_support_status(settings)
    if support == "unsupported":
        if best_text is not None:
            warnings.append(f"无法自动补充画面：{support_message}")
            return SmartReadResult(best_text, "已保留当前最完整的文字结果，但画面补充未执行。", warnings)
        raise RuntimeError("字幕和音频都不可用，且当前模型不能读取图片：" + support_message)

    try:
        progress(72, "有效文字仍不足，正在自动下载视频并补充画面信息……")
        video_path = download_video(part, auth_source)

        def visual_progress(percent: int, message: str) -> None:
            progress(max(72, min(99, 72 + percent * 27 // 100)), message)

        visual = analyze_video_frames(
            video_path,
            video_id=str(part["id"]),
            title=str(part["title"]),
            duration=duration,
            settings=settings,
            progress=visual_progress,
        )
    except Exception as exc:
        if best_text is not None:
            warnings.append(f"画面补充失败：{exc}")
            return SmartReadResult(best_text, "已保留当前最完整的文字结果。", warnings)
        raise

    combined = merge_text_and_visual(best_text, visual)
    message = "有效文字不足，已自动合并文字与视频画面信息。" if best_text else "已采用视频画面分析。"
    return SmartReadResult(combined, message, warnings, used_visual=True)
