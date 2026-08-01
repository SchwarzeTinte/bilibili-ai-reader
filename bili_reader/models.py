from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Segment":
        return cls(
            start=float(value.get("start", 0)),
            end=float(value.get("end", value.get("start", 0))),
            text=str(value.get("text", "")).strip(),
        )


@dataclass(slots=True)
class Transcript:
    video_id: str
    title: str
    source: str
    language: str
    segments: list[Segment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "source": self.source,
            "language": self.language,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Transcript":
        return cls(
            video_id=str(value["video_id"]),
            title=str(value.get("title", value["video_id"])),
            source=str(value.get("source", "未知")),
            language=str(value.get("language", "unknown")),
            segments=[Segment.from_dict(item) for item in value.get("segments", [])],
        )
