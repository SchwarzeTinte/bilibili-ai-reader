from __future__ import annotations

import html
import re

from .models import Segment


TIMESTAMP_RE = re.compile(
    r"(?P<start>(?:\d{1,2}:)?\d{1,2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>(?:\d{1,2}:)?\d{1,2}:\d{2}[,.]\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")


def timestamp_to_seconds(value: str) -> float:
    pieces = value.replace(",", ".").split(":")
    if len(pieces) == 2:
        hours = 0
        minutes, seconds = pieces
    elif len(pieces) == 3:
        hours, minutes, seconds = pieces
    else:
        raise ValueError(f"无法识别时间戳：{value}")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def clean_caption(value: str) -> str:
    value = value.replace("\\N", " ").replace("\u200b", " ")
    value = TAG_RE.sub("", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def merge_duplicate_segments(segments: list[Segment]) -> list[Segment]:
    merged: list[Segment] = []
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        text = clean_caption(segment.text)
        if not text:
            continue
        current = Segment(segment.start, max(segment.start, segment.end), text)
        if merged and merged[-1].text == current.text and current.start <= merged[-1].end + 0.5:
            merged[-1].end = max(merged[-1].end, current.end)
        else:
            merged.append(current)
    return merged


def parse_srt_or_vtt(content: str) -> list[Segment]:
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    segments: list[Segment] = []
    blocks = re.split(r"\n\s*\n", content)
    for block in blocks:
        lines = [line.strip("\ufeff ") for line in block.split("\n") if line.strip()]
        timing_index = next((i for i, line in enumerate(lines) if TIMESTAMP_RE.search(line)), None)
        if timing_index is None:
            continue
        match = TIMESTAMP_RE.search(lines[timing_index])
        if not match:
            continue
        text = clean_caption(" ".join(lines[timing_index + 1 :]))
        if text:
            segments.append(
                Segment(
                    timestamp_to_seconds(match.group("start")),
                    timestamp_to_seconds(match.group("end")),
                    text,
                )
            )
    return merge_duplicate_segments(segments)


def transcript_as_text(segments: list[Segment]) -> str:
    return "\n".join(f"[{format_timestamp(item.start)}] {item.text}" for item in segments)
