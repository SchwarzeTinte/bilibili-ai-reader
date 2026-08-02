from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import Transcript


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
TRASH_RETENTION_DAYS = 15


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


def history_path() -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return DATA_ROOT / "history.json"


def settings_path() -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return DATA_ROOT / "settings.json"


def load_app_settings() -> dict[str, Any]:
    """Load this installation's preferences without ever relying on machine paths."""
    path = settings_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def save_app_settings(settings: dict[str, Any]) -> Path:
    """Atomically persist preferences under the git-ignored local data directory."""
    path = settings_path()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def trash_directory(item_id: str) -> Path:
    return DATA_ROOT / ".trash" / safe_name(item_id)


def trash_content_directory(item_id: str) -> Path:
    return trash_directory(item_id) / "content"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_history() -> list[dict[str, Any]]:
    path = history_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [item for item in payload if isinstance(item, dict) and item.get("id")]
    except (OSError, ValueError, TypeError):
        return []


def _write_history(items: list[dict[str, Any]]) -> None:
    path = history_path()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _retry_pending_content_cleanup(items: list[dict[str, Any]]) -> bool:
    changed = False
    for item in items:
        if not item.get("content_cleanup_pending"):
            continue
        video_id = str(item.get("video_id", ""))
        active_content = DATA_ROOT / safe_name(video_id) if video_id else None
        still_referenced = any(
            other.get("id") != item.get("id")
            and not other.get("deleted_at")
            and str(other.get("video_id", "")) == video_id
            for other in items
        )
        if still_referenced:
            continue
        if active_content is not None and active_content.exists():
            try:
                shutil.rmtree(active_content)
            except OSError:
                continue
        item.pop("content_cleanup_pending", None)
        changed = True
    if changed:
        _write_history(items)
    return changed


def purge_expired_history(now: datetime | None = None) -> int:
    current = (now or _utc_now()).astimezone(timezone.utc)
    items = _read_history()
    kept: list[dict[str, Any]] = []
    expired_ids: list[str] = []
    for item in items:
        deleted_at = item.get("deleted_at")
        expires_at = _parse_datetime(item.get("expires_at"))
        if deleted_at and expires_at is not None and expires_at <= current:
            if item.get("content_cleanup_pending"):
                video_id = str(item.get("video_id", ""))
                active_content = DATA_ROOT / safe_name(video_id) if video_id else None
                still_referenced = any(
                    other.get("id") != item.get("id")
                    and not other.get("deleted_at")
                    and str(other.get("video_id", "")) == video_id
                    for other in items
                )
                if not still_referenced and active_content is not None and active_content.exists():
                    try:
                        shutil.rmtree(active_content)
                    except OSError:
                        kept.append(item)
                        continue
            expired_ids.append(str(item["id"]))
        else:
            kept.append(item)
    if not expired_ids:
        return 0
    _write_history(kept)
    for item_id in expired_ids:
        shutil.rmtree(trash_directory(item_id), ignore_errors=True)
    return len(expired_ids)


def list_history(include_archived: bool = False) -> list[dict[str, Any]]:
    purge_expired_history()
    items = _read_history()
    _retry_pending_content_cleanup(items)
    items = [item for item in items if not item.get("deleted_at")]
    if not include_archived:
        items = [item for item in items if not item.get("archived", False)]
    return sorted(items, key=lambda item: str(item.get("created_at", "")), reverse=True)


def list_deleted_history(now: datetime | None = None) -> list[dict[str, Any]]:
    purge_expired_history(now)
    all_items = _read_history()
    _retry_pending_content_cleanup(all_items)
    items = [item for item in all_items if item.get("deleted_at")]
    return sorted(items, key=lambda item: str(item.get("deleted_at", "")), reverse=True)


def add_history_item(
    *,
    transcript: Transcript,
    kind: str,
    title: str,
    content: str,
    provider: str,
    model: str,
    chat_history: list[dict[str, str]] | None = None,
    processing_seconds: float | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": uuid4().hex,
        "video_id": transcript.video_id,
        "video_title": transcript.title,
        "kind": kind,
        "title": title.strip() or transcript.title,
        "content": content,
        "provider": provider,
        "model": model,
        "chat_history": chat_history or [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archived": False,
    }
    if processing_seconds is not None:
        item["processing_seconds"] = max(0.0, float(processing_seconds))
    items = _read_history()
    items.append(item)
    _write_history(items)
    return item


def _merge_chat_histories(*histories: object) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for history in histories:
        if not isinstance(history, list):
            continue
        for entry in history:
            if not isinstance(entry, dict):
                continue
            question = str(entry.get("question", "")).strip()
            answer = str(entry.get("answer", "")).strip()
            if not question and not answer:
                continue
            identity = (question, answer)
            if identity in seen:
                continue
            seen.add(identity)
            merged.append({"question": question, "answer": answer})
    return merged


def consolidate_question_history_items() -> int:
    """Fold legacy per-question sidebar entries into one video conversation."""
    items = _read_history()
    video_ids = {
        str(item.get("video_id", ""))
        for item in items
        if item.get("kind") == "qa" and not item.get("deleted_at")
    }
    removed = 0
    for video_id in video_ids:
        related = [
            item
            for item in items
            if str(item.get("video_id", "")) == video_id and not item.get("deleted_at")
        ]
        if not related:
            continue
        non_question = [item for item in related if item.get("kind") != "qa"]
        candidates = [item for item in non_question if not item.get("archived")]
        target = max(
            candidates or non_question or related,
            key=lambda item: str(item.get("created_at", "")),
        )
        ordered = sorted(related, key=lambda item: str(item.get("created_at", "")))
        target["chat_history"] = _merge_chat_histories(
            *(item.get("chat_history") for item in ordered)
        )
        target["kind"] = "video"
        target["title"] = str(target.get("video_title") or target.get("title") or "未命名视频")
        redundant_ids = {
            str(item.get("id"))
            for item in related
            if item is not target and item.get("kind") == "qa"
        }
        if redundant_ids:
            items = [item for item in items if str(item.get("id")) not in redundant_ids]
            removed += len(redundant_ids)
    if video_ids:
        _write_history(items)
    return removed


def save_video_history_item(
    *,
    transcript: Transcript,
    provider: str,
    model: str,
    summary_content: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
    chat_branches: list[dict[str, Any]] | None = None,
    active_chat_branch: str = "",
    processing_seconds: float | None = None,
    preferred_item_id: str = "",
) -> dict[str, Any]:
    """Save notes and Q&A inside a video-level sidebar conversation."""
    items = _read_history()
    related = [
        item
        for item in items
        if str(item.get("video_id", "")) == transcript.video_id
        and not item.get("deleted_at")
    ]
    target = next(
        (
            item
            for item in related
            if str(item.get("id", "")) == preferred_item_id and item.get("kind") != "qa"
        ),
        None,
    )
    if target is None:
        non_question = [item for item in related if item.get("kind") != "qa"]
        visible = [item for item in non_question if not item.get("archived")]
        if visible or non_question:
            target = max(
                visible or non_question,
                key=lambda item: str(item.get("created_at", "")),
            )
    if target is None and related:
        target = max(related, key=lambda item: str(item.get("created_at", "")))
    if target is None:
        target = {
            "id": uuid4().hex,
            "video_id": transcript.video_id,
            "video_title": transcript.title,
            "kind": "video",
            "title": transcript.title,
            "content": "",
            "provider": provider,
            "model": model,
            "chat_history": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "archived": False,
        }
        items.append(target)
        related = [target]

    ordered = sorted(related, key=lambda item: str(item.get("created_at", "")))
    if chat_branches is None:
        target["chat_history"] = _merge_chat_histories(
            *(item.get("chat_history") for item in ordered),
            chat_history,
        )
    else:
        # Branch-aware callers provide the exact visible conversation. Merging
        # old snapshots here would re-add messages intentionally replaced by an edit.
        target["chat_history"] = list(chat_history or [])
        target["chat_branches"] = chat_branches
        target["active_chat_branch"] = active_chat_branch
    target.update(
        {
            "video_title": transcript.title,
            "kind": "video",
            "title": transcript.title,
            "provider": provider,
            "model": model,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    if summary_content is not None:
        target["content"] = summary_content
    if processing_seconds is not None:
        key = "processing_seconds" if summary_content is not None else "last_qa_seconds"
        target[key] = max(0.0, float(processing_seconds))

    redundant_ids = {
        str(item.get("id"))
        for item in related
        if item is not target and item.get("kind") == "qa"
    }
    if redundant_ids:
        items = [item for item in items if str(item.get("id")) not in redundant_ids]
    _write_history(items)
    return target


def update_history_item(item_id: str, **updates: Any) -> dict[str, Any] | None:
    items = _read_history()
    updated: dict[str, Any] | None = None
    for item in items:
        if item.get("id") == item_id:
            item.update(updates)
            updated = item
            break
    if updated is not None:
        _write_history(items)
    return updated


def soft_delete_history_item(
    item_id: str,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    items = _read_history()
    target = next((item for item in items if item.get("id") == item_id), None)
    if target is None:
        return None
    if target.get("deleted_at"):
        return target

    current = (now or _utc_now()).astimezone(timezone.utc)
    video_id = str(target.get("video_id", ""))
    active_content = DATA_ROOT / safe_name(video_id) if video_id else None
    backup_root = trash_directory(item_id)
    backup_content = trash_content_directory(item_id)
    remaining_references = [
        item
        for item in items
        if item.get("id") != item_id
        and not item.get("deleted_at")
        and str(item.get("video_id", "")) == video_id
    ]
    copied_content = False
    cleanup_pending = False
    if active_content is not None and active_content.exists():
        backup_root.mkdir(parents=True, exist_ok=True)
        # Copy first so a Windows file lock can never leave a half-moved backup.
        shutil.copytree(active_content, backup_content, dirs_exist_ok=True)
        copied_content = True
        if not remaining_references:
            try:
                shutil.rmtree(active_content)
            except OSError:
                # Open download/log files cannot be deleted on Windows. Keep the
                # recoverable backup and retry removing the active copy on later reruns.
                cleanup_pending = True

    target["deleted_at"] = current.isoformat()
    target["expires_at"] = (
        current + timedelta(days=TRASH_RETENTION_DAYS)
    ).isoformat()
    target["content_backup"] = copied_content
    if cleanup_pending:
        target["content_cleanup_pending"] = True
    else:
        target.pop("content_cleanup_pending", None)
    try:
        _write_history(items)
    except Exception:
        if copied_content and active_content is not None and backup_content.exists():
            active_content.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(backup_content, active_content, dirs_exist_ok=True)
            shutil.rmtree(backup_root, ignore_errors=True)
        raise
    return target


def restore_deleted_history_item(item_id: str) -> dict[str, Any] | None:
    items = _read_history()
    target = next((item for item in items if item.get("id") == item_id), None)
    if target is None or not target.get("deleted_at"):
        return None

    video_id = str(target.get("video_id", ""))
    active_content = DATA_ROOT / safe_name(video_id) if video_id else None
    backup_root = trash_directory(item_id)
    backup_content = trash_content_directory(item_id)
    copied_content = False
    if backup_content.exists() and active_content is not None:
        active_content.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(backup_content, active_content, dirs_exist_ok=True)
        copied_content = True

    previous = {
        "deleted_at": target.pop("deleted_at", None),
        "expires_at": target.pop("expires_at", None),
        "content_backup": target.pop("content_backup", None),
        "content_cleanup_pending": target.pop("content_cleanup_pending", None),
    }
    try:
        _write_history(items)
    except Exception:
        target.update({key: value for key, value in previous.items() if value is not None})
        raise
    if copied_content:
        shutil.rmtree(backup_root, ignore_errors=True)
    return target


def delete_history_item(item_id: str) -> bool:
    items = _read_history()
    kept = [item for item in items if item.get("id") != item_id]
    if len(kept) == len(items):
        return False
    _write_history(kept)
    shutil.rmtree(trash_directory(item_id), ignore_errors=True)
    return True
