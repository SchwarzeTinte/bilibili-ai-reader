from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Transcript


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned[:100] or "video"


def video_directory(video_id: str) -> Path:
    path = DATA_ROOT / safe_name(video_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def transcript_path(video_id: str) -> Path:
    return video_directory(video_id) / "transcript.json"


def save_transcript(transcript: Transcript) -> Path:
    path = transcript_path(transcript.video_id)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def load_transcript(video_id: str) -> Transcript | None:
    path = transcript_path(video_id)
    if not path.exists():
        return None
    try:
        return Transcript.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        return None
