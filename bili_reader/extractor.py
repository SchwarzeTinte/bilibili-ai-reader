from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from .models import Segment, Transcript
from .storage import safe_name, video_directory
from .text import merge_duplicate_segments, parse_srt_or_vtt


BVID_RE = re.compile(r"(?i)(BV[0-9A-Za-z]+)")
LANGUAGE_PRIORITY = (
    "zh-CN",
    "zh-Hans",
    "zh-Hant",
    "zh",
    "ai-zh",
    "ai-zh-CN",
    "中文",
)


class BilibiliReaderError(RuntimeError):
    pass


class _QuietYDLLogger:
    """Keep yt-dlp away from Streamlit's redirected console streams."""

    def debug(self, message: str) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass


def _cookie_options(source: str | None) -> dict[str, Any]:
    if not source or source.lower() == "不使用":
        return {}
    if source.startswith("file:"):
        cookie_file = Path(source[5:]).expanduser().resolve()
        if not cookie_file.is_file():
            raise BilibiliReaderError(f"找不到 cookies.txt：{cookie_file}")
        return {"cookiefile": str(cookie_file)}
    browser = source.removeprefix("browser:").lower()
    return {"cookiesfrombrowser": (browser, None, None, None)}


def _base_options(auth_source: str | None) -> dict[str, Any]:
    return {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _QuietYDLLogger(),
        "ignoreerrors": False,
        "socket_timeout": 30,
        "retries": 3,
        **_cookie_options(auth_source),
    }


def _flatten_entries(info: dict[str, Any]) -> list[dict[str, Any]]:
    entries = info.get("entries")
    if not entries:
        return [info]
    flattened: list[dict[str, Any]] = []
    for entry in entries:
        if entry:
            flattened.extend(_flatten_entries(entry))
    return flattened


def _download_url(original_url: str, index: int) -> str:
    match = BVID_RE.search(original_url)
    if not match:
        return original_url
    return f"https://www.bilibili.com/video/{match.group(1)}?p={index}"


def _subtitle_tracks(info: dict[str, Any]) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    for category_key, category_name in (
        ("subtitles", "字幕"),
        ("automatic_captions", "自动字幕"),
    ):
        for language, formats in (info.get(category_key) or {}).items():
            if language.lower() == "danmaku":
                continue
            for subtitle_format in formats or []:
                if subtitle_format.get("url") or subtitle_format.get("data"):
                    tracks.append(
                        {
                            "language": language,
                            "category": category_name,
                            "ext": subtitle_format.get("ext", "json"),
                            "url": subtitle_format.get("url"),
                            "data": subtitle_format.get("data"),
                            "name": subtitle_format.get("name") or language,
                        }
                    )
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for track in tracks:
        unique[(track["language"], track.get("url") or "inline")] = track

    def score(track: dict[str, Any]) -> tuple[int, int, str]:
        language = track["language"]
        try:
            priority = LANGUAGE_PRIORITY.index(language)
        except ValueError:
            priority = len(LANGUAGE_PRIORITY)
        automatic = int(language.lower().startswith("ai-") or track["category"] == "自动字幕")
        return automatic, priority, language

    return sorted(unique.values(), key=score)


def _has_audio_stream(info: dict[str, Any]) -> bool | None:
    """Infer audio availability from yt-dlp metadata without downloading media."""
    codecs: list[str] = []
    if info.get("acodec") is not None:
        codecs.append(str(info.get("acodec", "")).lower())
    formats = info.get("formats")
    if isinstance(formats, list):
        codecs.extend(
            str(item.get("acodec", "")).lower()
            for item in formats
            if isinstance(item, dict) and item.get("acodec") is not None
        )
    if any(codec and codec not in {"none", "null"} for codec in codecs):
        return True
    if codecs and all(codec in {"", "none", "null"} for codec in codecs):
        return False
    return None


def inspect_video(url: str, auth_source: str | None = None) -> list[dict[str, Any]]:
    if "bilibili.com" not in urlparse(url).netloc.lower() and not BVID_RE.search(url):
        raise BilibiliReaderError("请输入有效的哔哩哔哩视频链接或 BV 号。")
    if BVID_RE.fullmatch(url.strip()):
        url = f"https://www.bilibili.com/video/{url.strip()}"
    try:
        with YoutubeDL({**_base_options(auth_source), "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as exc:
        raise BilibiliReaderError(f"读取视频信息失败：{exc}") from exc
    if not info:
        raise BilibiliReaderError("没有读取到视频信息。")

    parts: list[dict[str, Any]] = []
    entries = _flatten_entries(info)
    for index, entry in enumerate(entries, start=1):
        video_id = str(entry.get("id") or f"part-{index}")
        thumbnail = entry.get("thumbnail")
        if isinstance(thumbnail, str) and thumbnail.startswith("http://"):
            thumbnail = "https://" + thumbnail.removeprefix("http://")
        parts.append(
            {
                "id": video_id,
                "title": str(entry.get("title") or info.get("title") or video_id),
                "duration": float(entry.get("duration") or 0),
                "index": int(entry.get("playlist_index") or index),
                "webpage_url": _download_url(url, int(entry.get("playlist_index") or index)),
                "thumbnail": thumbnail,
                "tracks": _subtitle_tracks(entry),
                "has_audio": _has_audio_stream(entry),
                "http_headers": entry.get("http_headers") or {"Referer": "https://www.bilibili.com/"},
            }
        )
    return parts


def _segments_from_bilibili_json(value: Any) -> list[Segment]:
    if isinstance(value, dict):
        if isinstance(value.get("body"), list):
            body = value["body"]
        elif isinstance(value.get("data"), dict) and isinstance(value["data"].get("body"), list):
            body = value["data"]["body"]
        else:
            body = []
    else:
        body = []
    segments = []
    for item in body:
        if not isinstance(item, dict):
            continue
        text = item.get("content") or item.get("text") or ""
        segments.append(
            Segment(
                float(item.get("from", item.get("start", 0))),
                float(item.get("to", item.get("end", item.get("from", 0)))),
                str(text),
            )
        )
    return merge_duplicate_segments(segments)


def download_subtitle(part: dict[str, Any], track: dict[str, Any]) -> Transcript:
    raw_data = track.get("data")
    if raw_data is None:
        url = str(track.get("url") or "")
        if url.startswith("//"):
            url = "https:" + url
        if not url:
            raise BilibiliReaderError("字幕轨道没有可用的下载地址。")
        headers = {str(key): str(value) for key, value in (part.get("http_headers") or {}).items()}
        headers.setdefault("Referer", "https://www.bilibili.com/")
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            raw_data = response.text
        except requests.RequestException as exc:
            raise BilibiliReaderError(f"字幕下载失败：{exc}") from exc

    segments: list[Segment]
    if isinstance(raw_data, (dict, list)):
        segments = _segments_from_bilibili_json(raw_data)
    else:
        text = str(raw_data).lstrip("\ufeff")
        try:
            parsed = json.loads(text)
            segments = _segments_from_bilibili_json(parsed)
        except json.JSONDecodeError:
            segments = parse_srt_or_vtt(text)
    if not segments:
        raise BilibiliReaderError("字幕已返回，但没有解析出有效文字。")
    return Transcript(
        video_id=part["id"],
        title=part["title"],
        source=f"B站{track['category']}",
        language=track["language"],
        segments=segments,
    )


def download_audio(part: dict[str, Any], auth_source: str | None = None) -> Path:
    target_dir = video_directory(part["id"])
    base_name = safe_name(part["id"]) + "-audio"
    cached_audio = (
        target_dir / f"{base_name}.mp3",
        target_dir / f"{base_name}-fallback.mp3",
    )
    for cached_path in cached_audio:
        if cached_path.is_file() and cached_path.stat().st_size > 0:
            return cached_path

    attempts = (
        (base_name, "bestaudio/best"),
        (f"{base_name}-fallback", "bestaudio[abr<=96]/bestaudio/best"),
    )
    errors: list[str] = []
    for attempt_name, format_selector in attempts:
        output_template = str(target_dir / f"{attempt_name}.%(ext)s")
        options = {
            **_base_options(auth_source),
            "format": format_selector,
            "noplaylist": True,
            "outtmpl": output_template,
            # Some Bilibili CDN nodes close long responses early. Small HTTP
            # ranges plus resume make those intermittent disconnects harmless.
            "http_chunk_size": 1024 * 1024,
            "retries": 10,
            "fragment_retries": 10,
            "file_access_retries": 3,
            "continuedl": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "128",
                }
            ],
        }
        try:
            with YoutubeDL(options) as ydl:
                ydl.download([part["webpage_url"]])
        except DownloadError as exc:
            errors.append(str(exc))
            continue

        expected = target_dir / f"{attempt_name}.mp3"
        if expected.exists():
            return expected
        candidates = sorted(
            (
                path
                for path in target_dir.glob(f"{attempt_name}.*")
                if not path.name.endswith((".part", ".ytdl"))
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
        errors.append("下载命令已结束，但没有找到生成的音频文件。")

    detail = errors[-1] if errors else "未知下载错误"
    raise BilibiliReaderError(f"音频下载失败（已尝试分块续传和低码率回退）：{detail}")


def download_video(part: dict[str, Any], auth_source: str | None = None) -> Path:
    """Download a moderate-resolution copy for local frame sampling."""
    target_dir = video_directory(part["id"])
    base_name = safe_name(part["id"]) + "-video"
    cached = sorted(
        (
            path
            for path in target_dir.glob(f"{base_name}.*")
            if path.is_file() and not path.name.endswith((".part", ".ytdl"))
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if cached:
        return cached[0]

    output_template = str(target_dir / f"{base_name}.%(ext)s")
    options = {
        **_base_options(auth_source),
        "format": "bestvideo[height<=720]/best[height<=720]/bestvideo/best",
        "noplaylist": True,
        "outtmpl": output_template,
        "http_chunk_size": 1024 * 1024,
        "retries": 10,
        "fragment_retries": 10,
        "file_access_retries": 3,
        "continuedl": True,
    }
    try:
        with YoutubeDL(options) as ydl:
            ydl.download([part["webpage_url"]])
    except DownloadError as exc:
        raise BilibiliReaderError(f"下载画面分析所需的视频失败：{exc}") from exc

    candidates = sorted(
        (
            path
            for path in target_dir.glob(f"{base_name}.*")
            if path.is_file() and not path.name.endswith((".part", ".ytdl"))
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise BilibiliReaderError("视频下载完成，但没有找到可供抽帧的文件。")
    return candidates[0]
