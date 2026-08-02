from __future__ import annotations

import re
import sys
import time
from math import ceil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bili_reader.extractor import (  # noqa: E402
    BilibiliReaderError,
    download_audio,
    download_subtitle,
    download_video,
    inspect_video,
)
from bili_reader.llm import (  # noqa: E402
    LLMSettings,
    answer_question,
    estimate_context_tokens,
    summarize,
    test_connection,
    vision_support_status,
)
from bili_reader.jobs import get_job_manager  # noqa: E402
from bili_reader.models import Transcript  # noqa: E402
from bili_reader.pipeline import SmartReadResult, smart_read_video  # noqa: E402
from bili_reader.storage import (  # noqa: E402
    consolidate_question_history_items,
    delete_history_item,
    list_deleted_history,
    list_history,
    load_app_settings,
    load_transcript,
    restore_deleted_history_item,
    save_transcript,
    save_app_settings,
    save_video_history_item,
    soft_delete_history_item,
    update_history_item,
)
from bili_reader.text import format_timestamp, transcript_as_text  # noqa: E402
from bili_reader.transcriber import transcribe_audio  # noqa: E402
from bili_reader.visual import (  # noqa: E402
    analyze_video_frames,
    frame_sampling_plan,
    vision_batch_size,
)


st.set_page_config(page_title="B站视频 AI 阅读器", page_icon="📺", layout="wide")


def workspace_from_url() -> str:
    raw = st.query_params.get("conversation", "")
    if isinstance(raw, list):
        raw = raw[-1] if raw else ""
    value = str(raw).strip()
    return value if re.fullmatch(r"[0-9A-Za-z_-]{1,80}", value) else ""


def remember_workspace_in_url(workspace_id: str) -> None:
    """Keep the visible conversation addressable across a browser refresh."""
    if workspace_id and workspace_from_url() != workspace_id:
        st.query_params["conversation"] = workspace_id


def initialize_state() -> None:
    fresh_session = "workspace_id" not in st.session_state
    if fresh_session:
        consolidate_question_history_items()
    initial_workspace_id = workspace_from_url() or uuid4().hex
    saved = load_app_settings()
    saved_llm = saved.get("llm", {}) if isinstance(saved.get("llm"), dict) else {}
    saved_reader = saved.get("reader", {}) if isinstance(saved.get("reader"), dict) else {}
    providers = {"Gemini", "DeepSeek", "OpenAI", "Anthropic", "OpenAI 兼容（自定义）", "Ollama"}
    provider = str(saved_llm.get("provider", "Gemini"))
    if provider not in providers:
        provider = "Gemini"
    try:
        context_window = max(2_048, int(saved_llm.get("context_window", 32_768)))
    except (TypeError, ValueError):
        context_window = 32_768
    initial_llm = LLMSettings(
        provider=provider,
        model=str(saved_llm.get("model", "gemini-2.5-flash")),
        api_key=str(saved_llm.get("api_key", "")),
        base_url=str(saved_llm.get("base_url", "")),
        context_window=context_window,
        max_context_window=saved_llm.get("max_context_window"),
    )
    initial_reader = {
        "auth_mode": str(saved_reader.get("auth_mode", "不使用")),
        "browser": str(saved_reader.get("browser", "Edge")),
        "cookie_path": str(saved_reader.get("cookie_path", "")),
        "whisper_model": str(saved_reader.get("whisper_model", "small")),
        "whisper_device": str(saved_reader.get("whisper_device", "CPU（兼容性优先）")),
        "whisper_language": str(saved_reader.get("whisper_language", "中文")),
        "visual_fallback_sensitivity": str(
            saved_reader.get("visual_fallback_sensitivity", "标准（推荐）")
        ),
    }
    if initial_reader["auth_mode"] not in {"不使用", "读取浏览器", "cookies.txt 文件"}:
        initial_reader["auth_mode"] = "不使用"
    if initial_reader["browser"] not in {"Edge", "Chrome", "Firefox"}:
        initial_reader["browser"] = "Edge"
    if initial_reader["whisper_model"] not in {"tiny", "base", "small", "medium", "large-v3"}:
        initial_reader["whisper_model"] = "small"
    if initial_reader["whisper_device"] not in {"CPU（兼容性优先）", "自动检测 GPU"}:
        initial_reader["whisper_device"] = "CPU（兼容性优先）"
    if initial_reader["whisper_language"] not in {"中文", "自动检测", "英文"}:
        initial_reader["whisper_language"] = "中文"
    if initial_reader["visual_fallback_sensitivity"] not in {
        "节省费用",
        "标准（推荐）",
        "严格完整",
    }:
        initial_reader["visual_fallback_sensitivity"] = "标准（推荐）"
    defaults = {
        "parts": [],
        "transcript": None,
        "summary": "",
        "chat_history": [],
        "video_summaries": {},
        "video_chats": {},
        "video_chat_branches": {},
        "video_active_chat_branches": {},
        "qa_editing": {},
        "pending_qa_edit": None,
        "qa_edit_errors": {},
        "current_video_id": "",
        "opened_history_id": "",
        "selected_history_id": "",
        "bulk_history_ids": [],
        "restore_history_id": "",
        "restore_dialog_open": False,
        "llm_settings": initial_llm,
        "reader_settings": initial_reader,
        "remember_api_key": bool(saved.get("remember_api_key", True)),
        "settings_dialog_open": False,
        "settings_error_message": "",
        "ai_error": None,
        "background_ai_jobs": {},
        "background_ai_errors": [],
        "background_media_jobs": {},
        "background_media_completed": [],
        "background_media_errors": [],
        "terminated_jobs": [],
        "workspace_id": initial_workspace_id,
        "fresh_page_session": fresh_session,
        "flash_message": "",
        "active_view": "内容总结",
        "inspected_url": "",
        "video_url": "",
        "audio_failures": {},
        "visual_accuracy_notices": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_new_conversation() -> None:
    """Reset only the active workspace; saved history and downloaded files stay intact."""
    st.session_state.parts = []
    st.session_state.transcript = None
    st.session_state.summary = ""
    st.session_state.chat_history = []
    st.session_state.qa_editing = {}
    st.session_state.current_video_id = ""
    st.session_state.opened_history_id = ""
    st.session_state.selected_history_id = ""
    st.session_state.bulk_history_ids = []
    st.session_state.inspected_url = ""
    st.session_state.video_url = ""
    st.session_state.active_view = "内容总结"
    st.session_state.ai_error = None
    st.session_state.workspace_id = uuid4().hex
    remember_workspace_in_url(st.session_state.workspace_id)
    st.session_state.flash_message = "已新建对话；原有历史和本地文件没有被删除。"


def interaction_busy() -> bool:
    return False


RUNNING_JOB_STATUSES = {"queued", "running"}
ACTIVE_JOB_STATUSES = RUNNING_JOB_STATUSES | {"canceling"}
LONG_TASK_SECONDS = 120


def restore_background_job_state() -> None:
    """Reconnect a fresh Streamlit session to jobs owned by this app process."""
    manager = get_job_manager()
    active_snapshots = []
    for snapshot in manager.snapshots():
        metadata = snapshot.metadata
        if not metadata:
            continue
        group = str(metadata.get("job_group", ""))
        if group == "ai":
            st.session_state.background_ai_jobs.setdefault(snapshot.job_id, metadata)
        elif group == "media":
            st.session_state.background_media_jobs.setdefault(snapshot.job_id, metadata)
        if snapshot.status in ACTIVE_JOB_STATUSES:
            active_snapshots.append(snapshot)

    if not st.session_state.fresh_page_session:
        return
    st.session_state.fresh_page_session = False
    requested_workspace = workspace_from_url()
    selected = next(
        (
            snapshot
            for snapshot in reversed(active_snapshots)
            if str((snapshot.metadata or {}).get("workspace_id", ""))
            == st.session_state.workspace_id
        ),
        None,
    )
    if selected is None and not requested_workspace and len(active_snapshots) == 1:
        selected = active_snapshots[0]
    if selected is not None and selected.metadata:
        open_running_conversation(selected.metadata, announce=False)
        st.session_state.flash_message = "页面刷新后已恢复正在运行的对话。"


def active_job_records(*, workspace_id: str | None = None) -> list[tuple[dict[str, object], object]]:
    """Return active jobs, optionally limited to one conversation workspace."""
    manager = get_job_manager()
    records: list[tuple[dict[str, object], object]] = []
    mappings = (
        st.session_state.background_ai_jobs,
        st.session_state.background_media_jobs,
    )
    for mapping in mappings:
        for job_id, stored_metadata in list(mapping.items()):
            snapshot = manager.snapshot(job_id)
            if snapshot is None or snapshot.status not in ACTIVE_JOB_STATUSES:
                continue
            metadata = snapshot.metadata or stored_metadata
            if workspace_id is not None and str(metadata.get("workspace_id")) != workspace_id:
                continue
            records.append((metadata, snapshot))
    return records


def job_uses_model(metadata: dict[str, object]) -> bool:
    return str(metadata.get("kind")) in {"summary", "qa", "smart", "visual"}


def model_runs_locally(settings: object) -> bool:
    if not isinstance(settings, LLMSettings):
        return False
    if settings.provider == "Ollama":
        return True
    return settings.provider == "OpenAI 兼容（自定义）" and any(
        host in settings.base_url.lower()
        for host in ("localhost", "127.0.0.1", "0.0.0.0")
    )


def visual_accuracy_notice(settings: object) -> tuple[str, bool]:
    if model_runs_locally(settings):
        return (
            "未检测到足够的有效文本，已改用本地模型识别视频画面。"
            "本地视觉模型的准确率可能受模型规模、量化方式和本机性能影响，"
            "可能低于大型云端视觉模型；请谨慎核对人物、画面文字、数字和关键情节。",
            True,
        )
    provider = settings.provider if isinstance(settings, LLMSettings) else "当前 API"
    model = settings.model if isinstance(settings, LLMSettings) else "所选模型"
    return (
        f"未检测到足够的有效文本，已改用 {provider} API 的 {model} 识别视频画面。"
        "生成准确率取决于该 API 实际使用模型的视觉能力，请结合原视频核对关键内容。",
        False,
    )


def current_model_jobs(*, lock_only: bool = False) -> list[tuple[dict[str, object], object]]:
    records = [
        record
        for record in active_job_records(workspace_id=str(st.session_state.workspace_id))
        if job_uses_model(record[0])
    ]
    if lock_only:
        records = [record for record in records if record[1].status in {"queued", "running"}]
    return records


def cancel_current_workspace_jobs(*, model_only: bool = False) -> int:
    manager = get_job_manager()
    records = active_job_records(workspace_id=str(st.session_state.workspace_id))
    canceled = 0
    for metadata, snapshot in records:
        if model_only and not job_uses_model(metadata):
            continue
        if manager.cancel(snapshot.job_id):
            canceled += 1
    return canceled


def forget_background_jobs(records: list[tuple[dict[str, object], object]]) -> int:
    """Remove canceled UI records while their worker discards any late result."""
    manager = get_job_manager()
    removed = 0
    for _, snapshot in records:
        manager.cancel(snapshot.job_id)
        manager.discard(snapshot.job_id)
        st.session_state.background_ai_jobs.pop(snapshot.job_id, None)
        st.session_state.background_media_jobs.pop(snapshot.job_id, None)
        removed += 1
    return removed


def job_title(metadata: dict[str, object]) -> str:
    source = metadata.get("transcript") or metadata.get("part")
    title = getattr(source, "title", "") if not isinstance(source, dict) else source.get("title", "")
    if title:
        return str(title)
    url = str(metadata.get("url", "")).strip()
    if url:
        bv_match = re.search(r"BV[0-9A-Za-z]+", url)
        return bv_match.group(0) if bv_match else "正在读取新视频"
    return "正在处理的视频"


def format_task_duration(seconds: float | int | None) -> str:
    total = max(0, int(seconds or 0))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def job_elapsed_seconds(snapshot) -> float:
    start = snapshot.started_at or snapshot.created_at or time.time()
    end = snapshot.cancel_requested_at or snapshot.finished_at or time.time()
    return max(0.0, end - start)


def job_timing_text(snapshot) -> str:
    elapsed = job_elapsed_seconds(snapshot)
    if snapshot.status == "queued":
        prefix = f"已排队 {format_task_duration(elapsed)}"
    elif snapshot.status == "canceling":
        prefix = f"终止前运行 {format_task_duration(elapsed)}"
    else:
        prefix = f"已运行 {format_task_duration(elapsed)}"
    estimate = snapshot.estimated_seconds
    if not estimate:
        return prefix
    if elapsed > estimate * 1.25:
        return f"{prefix} · 已超过粗略预计 {format_task_duration(estimate)}"
    return f"{prefix} · 粗略预计 {format_task_duration(estimate)}"


def estimate_media_job_seconds(
    kind: str,
    part: dict[str, object],
    settings: LLMSettings,
    reader: dict[str, object],
) -> tuple[float, str]:
    duration = max(1.0, float(part.get("duration") or 0))
    is_local_model = settings.provider == "Ollama" or (
        settings.provider == "OpenAI 兼容（自定义）"
        and any(host in settings.base_url.lower() for host in ("localhost", "127.0.0.1"))
    )
    if kind == "subtitle":
        return 20.0, "字幕下载与解析"
    whisper_model = str(reader.get("whisper_model", "small"))
    cpu_factor = {"tiny": 0.18, "base": 0.3, "small": 0.65, "medium": 1.1, "large-v3": 1.65}
    device_factor = (
        cpu_factor.get(whisper_model, 0.65)
        if reader.get("whisper_device") == "CPU（兼容性优先）"
        else max(0.12, cpu_factor.get(whisper_model, 0.65) * 0.28)
    )
    audio_estimate = 35.0 + duration * device_factor
    planned_frames, _ = frame_sampling_plan(duration)
    batches = ceil(planned_frames / vision_batch_size(settings))
    visual_estimate = 45.0 + batches * (14.0 if is_local_model else 10.0)
    if kind == "audio":
        return audio_estimate, f"按视频时长与 Whisper {whisper_model} 估算"
    if kind == "visual":
        return visual_estimate, f"按约 {planned_frames} 张抽帧估算"
    if kind in {"smart", "reread"}:
        return 35.0 + audio_estimate + visual_estimate * 0.35, "按字幕、音频及可能的画面补充估算"
    return 30.0, "按网络检测步骤估算"


def estimate_ai_job_seconds(
    kind: str, transcript: Transcript, settings: LLMSettings
) -> tuple[float, str]:
    text_chars = sum(len(segment.text) for segment in transcript.segments)
    is_local = settings.provider == "Ollama" or (
        settings.provider == "OpenAI 兼容（自定义）"
        and any(host in settings.base_url.lower() for host in ("localhost", "127.0.0.1"))
    )
    if is_local:
        base = 45.0 if kind == "qa" else 75.0
        estimate = base + text_chars / (110.0 if kind == "qa" else 70.0)
        return estimate, "按字幕长度与本地模型速度粗略估算"
    base = 12.0 if kind == "qa" else 25.0
    estimate = base + text_chars / (900.0 if kind == "qa" else 500.0)
    return estimate, "按字幕长度与云端接口往返粗略估算"


def remember_terminated_job(metadata: dict[str, object], snapshot=None) -> None:
    """Keep a removable UI record after an unfinished job has fully stopped."""
    job_id = str(metadata.get("job_id", ""))
    records = [
        record
        for record in st.session_state.terminated_jobs
        if str(record.get("job_id", "")) != job_id
    ]
    settings = metadata.get("settings")
    records.insert(
        0,
        {
            "job_id": job_id,
            "title": job_title(metadata),
            "kind": str(metadata.get("kind", "")),
            "provider": getattr(settings, "provider", ""),
            "model": getattr(settings, "model", ""),
            "elapsed_seconds": job_elapsed_seconds(snapshot) if snapshot is not None else None,
        },
    )
    st.session_state.terminated_jobs = records[:20]


def open_running_conversation(metadata: dict[str, object], *, announce: bool = True) -> None:
    """Switch the visible workspace without touching the worker that owns it."""
    st.session_state.workspace_id = str(metadata.get("workspace_id", st.session_state.workspace_id))
    remember_workspace_in_url(st.session_state.workspace_id)
    st.session_state.opened_history_id = ""
    st.session_state.selected_history_id = f"running:{st.session_state.workspace_id}"
    st.session_state.ai_error = None
    kind = str(metadata.get("kind", ""))
    transcript = metadata.get("transcript")
    if isinstance(transcript, Transcript):
        activate_transcript(transcript)
        if kind == "qa":
            history = list(metadata.get("history", []))
            st.session_state.video_chats[transcript.video_id] = history
            st.session_state.chat_history = history
            st.session_state.video_chat_branches[transcript.video_id] = list(
                metadata.get("chat_branches", [])
            )
            st.session_state.video_active_chat_branches[transcript.video_id] = str(
                metadata.get("active_chat_branch", "")
            )
            st.session_state.active_view = "视频问答"
        elif kind == "summary":
            st.session_state.active_view = "内容总结"
        st.session_state.parts = []
    elif kind == "inspect":
        st.session_state.video_url = str(metadata.get("url", ""))
        st.session_state.inspected_url = ""
        st.session_state.parts = []
        st.session_state.transcript = None
        st.session_state.summary = ""
        st.session_state.chat_history = []
        st.session_state.current_video_id = ""
    else:
        part = metadata.get("part")
        if isinstance(part, dict):
            st.session_state.parts = [part]
            cached = load_transcript(str(part.get("id", "")))
            if cached is not None:
                activate_transcript(cached)
    if announce:
        st.session_state.flash_message = "已返回正在运行的对话；后台任务没有中断。"


def auth_source_from_reader(reader: dict[str, object]) -> str | None:
    auth_mode = str(reader.get("auth_mode", "不使用"))
    if auth_mode == "读取浏览器":
        return f"browser:{reader.get('browser', 'Edge')}"
    if auth_mode == "cookies.txt 文件":
        cookie_path = str(reader.get("cookie_path", "")).strip().strip(chr(34))
        return f"file:{cookie_path}" if cookie_path else None
    return None


def schedule_media_job(kind: str, part: dict[str, object], **payload: object) -> None:
    remember_workspace_in_url(str(st.session_state.workspace_id))
    reader = dict(st.session_state.reader_settings)
    settings = st.session_state.llm_settings
    settings_snapshot = LLMSettings(
        provider=settings.provider,
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        context_window=settings.context_window,
        max_context_window=settings.max_context_window,
    )
    auth_source = auth_source_from_reader(reader)
    language_map = {"中文": "zh", "自动检测": None, "英文": "en"}
    whisper_language = language_map.get(str(reader.get("whisper_language", "中文")))
    whisper_device = (
        "cpu" if reader.get("whisper_device") == "CPU（兼容性优先）" else "auto"
    )

    def run(progress):
        if kind in {"smart", "reread"}:
            target_part = part
            if kind == "reread":
                progress(2, "正在重新检测视频信息、字幕和音轨……")
                detected_parts = inspect_video(str(part.get("url") or part["id"]), auth_source)
                if not detected_parts:
                    raise RuntimeError("重新检测后没有找到可读取的分P。")
                target_part = next(
                    (
                        detected
                        for detected in detected_parts
                        if str(detected.get("id", "")) == str(part.get("id", ""))
                    ),
                    detected_parts[0],
                )
            return smart_read_video(
                target_part,
                auth_source=auth_source,
                whisper_model=str(reader.get("whisper_model", "small")),
                whisper_language=whisper_language,
                whisper_device=whisper_device,
                settings=settings_snapshot,
                progress=progress,
                visual_fallback_sensitivity=str(
                    reader.get("visual_fallback_sensitivity", "标准（推荐）")
                ),
            )
        if kind == "subtitle":
            progress(10, "正在下载并解析字幕……")
            transcript = download_subtitle(part, payload["track"])
            return SmartReadResult(transcript, "字幕提取完成并已缓存到本机。")
        if kind == "audio":
            progress(2, "正在下载音频……")
            audio_path = download_audio(part, auth_source)
            transcript = transcribe_audio(
                audio_path,
                video_id=str(part["id"]),
                title=str(part["title"]),
                model_size=str(reader.get("whisper_model", "small")),
                language=whisper_language,
                device_mode=whisper_device,
                progress=progress,
            )
            return SmartReadResult(transcript, "音频转写完成并已缓存到本机。")
        if kind == "visual":
            progress(2, "正在下载画面分析所需的视频……")
            video_path = download_video(part, auth_source)
            transcript = analyze_video_frames(
                video_path,
                video_id=str(part["id"]),
                title=str(part["title"]),
                duration=float(part.get("duration") or 0),
                settings=settings_snapshot,
                progress=progress,
            )
            return SmartReadResult(
                transcript,
                "画面分析完成，可以继续生成笔记或问答。",
                used_visual=True,
            )
        raise RuntimeError("未知的媒体处理任务。")

    manager = get_job_manager()
    job_id = manager.submit(run, "媒体任务正在后台启动……")
    is_local = kind in {"audio", "smart", "reread"} or settings.provider == "Ollama"
    metadata = {
        "job_id": job_id,
        "job_group": "media",
        "kind": kind,
        "part": part,
        "settings": settings_snapshot,
        "workspace_id": st.session_state.workspace_id,
        "local": is_local,
    }
    manager.set_metadata(job_id, metadata)
    estimated_seconds, estimate_note = estimate_media_job_seconds(
        kind, part, settings_snapshot, reader
    )
    manager.set_estimate(job_id, estimated_seconds, estimate_note)
    st.session_state.background_media_jobs[job_id] = metadata
    label = JOB_KIND_LABELS.get(kind, "视频读取")
    if estimated_seconds >= LONG_TASK_SECONDS:
        st.session_state.flash_message = (
            f"{label}预计约 {format_task_duration(estimated_seconds)}，可能需要较长时间；"
            "任务已在后台运行。"
        )
    else:
        st.session_state.flash_message = f"{label}已在后台开始。"
    st.rerun()


def schedule_video_inspection(url: str) -> None:
    auth_source = auth_source_from_reader(dict(st.session_state.reader_settings))
    inspection_workspace_id = uuid4().hex
    st.session_state.workspace_id = inspection_workspace_id
    remember_workspace_in_url(inspection_workspace_id)
    manager = get_job_manager()
    job_id = manager.submit(
        lambda progress: inspect_video(url.strip(), auth_source),
        "正在读取视频信息、字幕和音轨列表……",
    )
    metadata = {
        "job_id": job_id,
        "job_group": "media",
        "kind": "inspect",
        "url": url.strip(),
        "workspace_id": inspection_workspace_id,
        "local": False,
    }
    manager.set_metadata(job_id, metadata)
    manager.set_estimate(job_id, 30.0, "按B站网络请求与视频信息解析估算")
    st.session_state.background_media_jobs[job_id] = metadata
    st.session_state.flash_message = "视频信息读取已在后台开始。"
    st.rerun()


def activate_transcript(transcript: Transcript) -> None:
    video_id = transcript.video_id
    st.session_state.transcript = transcript
    st.session_state.current_video_id = video_id
    st.session_state.summary = st.session_state.video_summaries.get(video_id, "")
    st.session_state.chat_history = list(st.session_state.video_chats.get(video_id, []))


def normalize_chat_branches(
    video_id: str,
    history: list[dict[str, str]],
) -> tuple[list[dict[str, object]], str]:
    """Return branch snapshots, upgrading a legacy linear chat in memory."""
    raw = st.session_state.video_chat_branches.get(video_id, [])
    branches = [dict(branch) for branch in raw if isinstance(branch, dict)]
    if not branches:
        branches = [
            {
                "id": f"main-{uuid4().hex}",
                "history": list(history),
                "fork_index": None,
            }
        ]
    active_id = str(st.session_state.video_active_chat_branches.get(video_id, ""))
    if not any(str(branch.get("id")) == active_id for branch in branches):
        active_id = str(branches[-1]["id"])
    # Keep the in-memory upgrade stable across Streamlit reruns. Without this,
    # a legacy linear chat receives a new temporary branch id on every click,
    # making a valid edited question look stale before it can be submitted.
    st.session_state.video_chat_branches[video_id] = branches
    st.session_state.video_active_chat_branches[video_id] = active_id
    return branches, active_id


def select_chat_branch(video_id: str, branch_id: str) -> None:
    branches, _ = normalize_chat_branches(video_id, st.session_state.chat_history)
    selected = next(
        (branch for branch in branches if str(branch.get("id")) == branch_id),
        None,
    )
    if selected is None:
        return
    history = list(selected.get("history", []))
    st.session_state.video_chat_branches[video_id] = branches
    st.session_state.video_active_chat_branches[video_id] = branch_id
    st.session_state.video_chats[video_id] = history
    st.session_state.chat_history = history
    st.session_state.qa_editing.pop(video_id, None)
    if st.session_state.opened_history_id:
        update_history_item(
            st.session_state.opened_history_id,
            chat_history=history,
            chat_branches=branches,
            active_chat_branch=branch_id,
        )


def begin_question_edit(video_id: str, index: int) -> None:
    """Use an assignment so Streamlit reliably observes the edit state."""
    editing = dict(st.session_state.get("qa_editing", {}))
    editing[video_id] = index
    st.session_state.qa_editing = editing


def cancel_question_edit(video_id: str) -> None:
    editing = dict(st.session_state.get("qa_editing", {}))
    editing.pop(video_id, None)
    st.session_state.qa_editing = editing
    errors = dict(st.session_state.get("qa_edit_errors", {}))
    errors.pop(video_id, None)
    st.session_state.qa_edit_errors = errors
    pending = st.session_state.get("pending_qa_edit")
    if isinstance(pending, dict) and pending.get("video_id") == video_id:
        st.session_state.pending_qa_edit = None


def queue_question_edit(
    video_id: str,
    index: int,
    branch_id: str,
    widget_key: str,
) -> None:
    """Capture the form value before Streamlit rebuilds the question list."""
    question = str(st.session_state.get(widget_key, "")).strip()
    errors = dict(st.session_state.get("qa_edit_errors", {}))
    if not question:
        errors[video_id] = "问题不能为空。"
        st.session_state.qa_edit_errors = errors
        return
    errors.pop(video_id, None)
    st.session_state.qa_edit_errors = errors
    st.session_state.pending_qa_edit = {
        "video_id": video_id,
        "index": index,
        "branch_id": branch_id,
        "question": question,
    }


def open_history_item(item: dict[str, object]) -> bool:
    video_id = str(item.get("video_id", ""))
    transcript = load_transcript(video_id)
    if transcript is None:
        st.sidebar.error("这条历史记录对应的本地字幕已经不存在，无法重新打开。")
        return False
    kind = str(item.get("kind", "summary"))
    if kind in {"summary", "video"}:
        content = str(item.get("content", ""))
        st.session_state.video_summaries[video_id] = content
        chat_history = item.get("chat_history", [])
        if not isinstance(chat_history, list):
            chat_history = []
        st.session_state.video_chats[video_id] = chat_history
        branches = item.get("chat_branches", [])
        st.session_state.video_chat_branches[video_id] = (
            branches if isinstance(branches, list) else []
        )
        st.session_state.video_active_chat_branches[video_id] = str(
            item.get("active_chat_branch", "")
        )
        st.session_state.active_view = "内容总结" if content else "视频问答"
    else:
        chat_history = item.get("chat_history", [])
        if not isinstance(chat_history, list):
            chat_history = []
        st.session_state.video_chats[video_id] = chat_history
        branches = item.get("chat_branches", [])
        st.session_state.video_chat_branches[video_id] = (
            branches if isinstance(branches, list) else []
        )
        st.session_state.video_active_chat_branches[video_id] = str(
            item.get("active_chat_branch", "")
        )
        st.session_state.active_view = "视频问答"
    st.session_state.opened_history_id = str(item.get("id", ""))
    st.session_state.workspace_id = uuid4().hex
    remember_workspace_in_url(st.session_state.workspace_id)
    activate_transcript(transcript)
    st.session_state.parts = []
    return True


def compact_history_title(item: dict[str, object], max_length: int = 14) -> str:
    if item.get("kind") in {"summary", "video"}:
        raw_title = str(item.get("video_title") or item.get("title") or "未命名视频")
    else:
        raw_title = str(item.get("title") or "未命名问答")
    title = re.sub(r"\s+", " ", raw_title).strip()
    if len(title) <= max_length:
        return title
    return title[: max_length - 1].rstrip() + "…"


def clear_deleted_history_item(item_id: str, item: dict[str, object]) -> None:
    if st.session_state.opened_history_id != item_id:
        return
    video_id = str(item.get("video_id", ""))
    if item.get("kind") == "video":
        st.session_state.video_summaries.pop(video_id, None)
        st.session_state.video_chats.pop(video_id, None)
        st.session_state.video_chat_branches.pop(video_id, None)
        st.session_state.video_active_chat_branches.pop(video_id, None)
        st.session_state.qa_editing.pop(video_id, None)
        st.session_state.summary = ""
        st.session_state.chat_history = []
    elif item.get("kind") == "summary":
        st.session_state.video_summaries.pop(video_id, None)
        st.session_state.summary = ""
    else:
        st.session_state.video_chats.pop(video_id, None)
        st.session_state.chat_history = []
    if st.session_state.current_video_id == video_id:
        st.session_state.transcript = None
        st.session_state.current_video_id = ""
        st.session_state.parts = []
    st.session_state.opened_history_id = ""


def set_bulk_history_selection(item_ids: list[str]) -> None:
    st.session_state.pending_bulk_history_ids = list(item_ids)


def archive_bulk_history(item_ids: list[str], archived: bool) -> None:
    for item_id in item_ids:
        update_history_item(item_id, archived=archived)
    st.session_state.pending_bulk_history_ids = []


def delete_bulk_history(
    item_ids: list[str],
    item_by_id: dict[str, dict[str, object]],
) -> None:
    deleted_count = 0
    pending_cleanup = 0
    failures: list[str] = []
    for item_id in item_ids:
        item = item_by_id.get(item_id)
        if item is None:
            continue
        try:
            deleted = soft_delete_history_item(item_id)
        except OSError as exc:
            failures.append(f"{item.get('title', '未命名记录')}：{exc}")
            continue
        if deleted is None:
            continue
        clear_deleted_history_item(item_id, item)
        deleted_count += 1
        if deleted.get("content_cleanup_pending"):
            pending_cleanup += 1
    st.session_state.selected_history_id = ""
    st.session_state.pending_bulk_history_ids = []
    if failures:
        st.session_state.flash_message = (
            f"已删除 {deleted_count} 条；另有 {len(failures)} 条因文件无法读取而未删除。"
        )
    elif pending_cleanup:
        st.session_state.flash_message = (
            f"已移入15天回收站；{pending_cleanup} 个正在占用的文件将在任务释放后自动清理。"
        )
    else:
        st.session_state.flash_message = f"已将 {deleted_count} 条记录移入15天回收站。"


def deleted_history_countdown(item: dict[str, object]) -> str:
    try:
        expires_at = datetime.fromisoformat(str(item.get("expires_at", "")))
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
    except ValueError:
        return "恢复期限未知"
    remaining = (expires_at.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        return "即将自动清理"
    total_hours = max(1, int((remaining + 3_599) // 3_600))
    days, hours = divmod(total_hours, 24)
    if days and hours:
        return f"{days}天{hours}小时后自动清理"
    if days:
        return f"{days}天后自动清理"
    return f"{hours}小时后自动清理"


def open_restore_dialog(item_id: str) -> None:
    st.session_state.restore_history_id = item_id
    st.session_state.restore_dialog_open = True


def dismiss_restore_dialog() -> None:
    st.session_state.restore_dialog_open = False


@st.dialog("恢复已删除内容", on_dismiss=dismiss_restore_dialog)
def restore_history_dialog() -> None:
    item_id = str(st.session_state.restore_history_id)
    item = next(
        (entry for entry in list_deleted_history() if str(entry.get("id")) == item_id),
        None,
    )
    if item is None:
        st.warning("这条记录已经恢复或超过15天并被自动清理。")
        if st.button("关闭", use_container_width=True, key="close_missing_restore"):
            st.session_state.restore_dialog_open = False
            st.rerun()
        return
    st.markdown(f"**{item.get('title') or item.get('video_title') or '未命名内容'}**")
    st.info(f"⏳ {deleted_history_countdown(item)}")
    st.write("恢复后，聊天/总结记录以及备份的字幕、音频等本地文件会重新回到正常内容区。")
    restore_col, cancel_col = st.columns(2)
    if restore_col.button(
        "恢复全部内容",
        type="primary",
        use_container_width=True,
        key="confirm_restore_history",
    ):
        restored = restore_deleted_history_item(item_id)
        if restored is None:
            st.error("恢复失败：记录不存在或已经恢复。")
        else:
            st.session_state.restore_dialog_open = False
            st.session_state.restore_history_id = ""
            st.session_state.selected_history_id = item_id
            if load_transcript(str(restored.get("video_id", ""))) is not None:
                open_history_item(restored)
                st.session_state.flash_message = "聊天记录和本地视频内容已恢复。"
            else:
                st.session_state.flash_message = "聊天记录已恢复，但原记录没有可恢复的本地字幕文件。"
            st.rerun()
    if cancel_col.button(
        "暂不恢复",
        use_container_width=True,
        key="cancel_restore_history",
    ):
        st.session_state.restore_dialog_open = False
        st.rerun()
    st.divider()
    confirm_permanent_delete = st.checkbox(
        "我确认永久删除这条聊天及其回收站文件（无法恢复）",
        key=f"confirm_permanent_delete_{item_id}",
    )
    if st.button(
        "永久删除",
        type="secondary",
        use_container_width=True,
        key=f"permanently_delete_{item_id}",
        disabled=not confirm_permanent_delete,
    ):
        if delete_history_item(item_id):
            st.session_state.restore_dialog_open = False
            st.session_state.restore_history_id = ""
            st.session_state.flash_message = "该聊天及其回收站文件已永久删除，无法恢复。"
            st.rerun()
        st.error("永久删除失败：这条记录可能已经不存在。")


def bulk_history_manager(
    item_by_id: dict[str, dict[str, object]],
    visible_ids: list[str],
    disabled: bool,
) -> None:
    if "pending_bulk_history_ids" in st.session_state:
        st.session_state.bulk_history_ids = st.session_state.pop(
            "pending_bulk_history_ids"
        )
    current_bulk = [
        item_id
        for item_id in st.session_state.get("bulk_history_ids", [])
        if item_id in item_by_id
    ]
    if current_bulk != st.session_state.get("bulk_history_ids", []):
        st.session_state.bulk_history_ids = current_bulk
    with st.expander("管理历史记录"):
        bulk_ids = st.multiselect(
            "选择要批量处理的记录",
            options=visible_ids,
            format_func=lambda item_id: compact_history_title(item_by_id[item_id], 22),
            key="bulk_history_ids",
            placeholder="可选择多条记录",
            disabled=disabled,
        )
        select_col, clear_col = st.columns(2)
        select_col.button(
            "全选当前列表",
            key="select_all_history",
            use_container_width=True,
            disabled=disabled or len(bulk_ids) == len(visible_ids),
            on_click=set_bulk_history_selection,
            args=(visible_ids,),
        )
        clear_col.button(
            "清空选择",
            key="clear_bulk_history",
            use_container_width=True,
            disabled=disabled or not bulk_ids,
            on_click=set_bulk_history_selection,
            args=([],),
        )
        archive_ids = [
            item_id for item_id in bulk_ids if not item_by_id[item_id].get("archived")
        ]
        restore_ids = [
            item_id for item_id in bulk_ids if item_by_id[item_id].get("archived")
        ]
        archive_col, restore_col = st.columns(2)
        archive_col.button(
            f"归档所选（{len(archive_ids)}）",
            key="archive_bulk_history",
            icon=":material/archive:",
            use_container_width=True,
            disabled=disabled or not archive_ids,
            on_click=archive_bulk_history,
            args=(archive_ids, True),
        )
        restore_col.button(
            f"取消归档（{len(restore_ids)}）",
            key="restore_bulk_history",
            use_container_width=True,
            disabled=disabled or not restore_ids,
            on_click=archive_bulk_history,
            args=(restore_ids, False),
        )
        confirm_bulk_delete = st.checkbox(
            "确认将全部所选记录移至已删除",
            key="confirm_bulk_history_delete",
            disabled=disabled or not bulk_ids,
        )
        st.caption("聊天/总结及关联本地文件会进入15天可恢复区，而不是立即永久删除。")
        st.button(
            f"移至已删除（{len(bulk_ids)}）",
            key="delete_bulk_history",
            icon=":material/delete:",
            use_container_width=True,
            disabled=disabled or not bulk_ids or not confirm_bulk_delete,
            on_click=delete_bulk_history,
            args=(bulk_ids, item_by_id),
        )


JOB_KIND_LABELS = {
    "summary": "详细笔记",
    "qa": "视频问答",
    "inspect": "检测视频",
    "smart": "智能读取",
    "reread": "重新读取",
    "subtitle": "提取字幕",
    "audio": "音频转写",
    "visual": "画面分析",
}


@st.fragment(run_every="1s")
def running_conversations_sidebar() -> None:
    records = active_job_records()
    if not records:
        return
    grouped: dict[str, list[tuple[dict[str, object], object]]] = {}
    for metadata, snapshot in records:
        workspace_id = str(metadata.get("workspace_id", snapshot.job_id))
        grouped.setdefault(workspace_id, []).append((metadata, snapshot))

    st.markdown(
        """
        <style>
        .running-progress-ring {
            width: 1.25rem;
            height: 1.25rem;
            margin: 0.42rem auto 0;
            border-radius: 50%;
            display: grid;
            place-items: center;
        }
        .running-progress-ring::after {
            content: "";
            width: 0.72rem;
            height: 0.72rem;
            border-radius: 50%;
            background: var(--secondary-background-color, #fff);
        }
        .running-progress-ring.is-running {
            animation: running-progress-spin 1.1s linear infinite;
        }
        @keyframes running-progress-spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.caption("后台任务")
    for workspace_id, jobs in grouped.items():
        snapshots = [snapshot for _, snapshot in jobs]
        running_snapshots = [
            snapshot for snapshot in snapshots if snapshot.status in RUNNING_JOB_STATUSES
        ]
        progress_source = running_snapshots or snapshots
        progress = round(
            sum(snapshot.progress for snapshot in progress_source) / len(progress_source)
        )
        is_running = bool(running_snapshots)
        display_metadata, _ = next(
            (
                record
                for record in jobs
                if record[1].status in RUNNING_JOB_STATUSES
            ),
            jobs[0],
        )
        display_progress = max(3, min(100, progress))
        title = job_title(display_metadata)
        status_text = "；".join(
            f"{JOB_KIND_LABELS.get(str(job_metadata.get('kind')), '后台任务')} · "
            f"{job_title(job_metadata)}：{snapshot.message}（{job_timing_text(snapshot)}）"
            for job_metadata, snapshot in jobs
        )
        title_col, progress_col = st.columns([0.86, 0.14], gap="small")
        with title_col:
            if st.button(
                compact_history_title({"title": title, "kind": "qa"}),
                key=f"running_conversation_{workspace_id}",
                type="tertiary",
                help=f"{title}\n{status_text}",
                use_container_width=True,
            ):
                open_running_conversation(display_metadata)
                st.rerun()
        with progress_col:
            if is_running:
                st.markdown(
                    f'<div class="running-progress-ring is-running" title="{progress}%" '
                    f'style="background:conic-gradient(#ff4b4b {display_progress}%, '
                    'rgba(128,128,128,.22) 0);"></div>',
                    unsafe_allow_html=True,
                )
            elif st.button(
                "×",
                key=f"forget_canceling_conversation_{workspace_id}",
                type="tertiary",
                help="删除已终止任务记录；任何迟到结果仍会被丢弃。",
            ):
                forget_background_jobs(jobs)
                st.rerun()


@st.fragment(run_every="1s")
def current_conversation_progress_panel() -> None:
    records = active_job_records(workspace_id=str(st.session_state.workspace_id))
    if not records:
        return
    running_records = [record for record in records if record[1].status in RUNNING_JOB_STATUSES]
    canceling_records = [record for record in records if record[1].status == "canceling"]
    with st.container(border=True):
        header_col, stop_col = st.columns([0.78, 0.22], vertical_alignment="center")
        with header_col:
            st.markdown("#### 当前对话后台任务")
        with stop_col:
            if running_records and st.button(
                "终止本对话全部任务",
                key="cancel_all_current_conversation_jobs",
                type="secondary",
                use_container_width=True,
            ):
                canceled = cancel_current_workspace_jobs()
                st.session_state.flash_message = f"已向当前对话的 {canceled} 个任务发出终止请求。"
                st.rerun()
        for metadata, snapshot in running_records:
            label = JOB_KIND_LABELS.get(str(metadata.get("kind")), "后台任务")
            st.caption(f"⏱️ {label} · {job_title(metadata)} · {job_timing_text(snapshot)}")
            st.progress(snapshot.progress, text=snapshot.message)
            kind = str(metadata.get("kind", ""))
            visual_phase = kind == "visual" or (
                kind in {"smart", "reread"}
                and ("画面" in snapshot.message or snapshot.progress >= 72)
            )
            if visual_phase:
                notice, is_local = visual_accuracy_notice(metadata.get("settings"))
                if is_local:
                    st.warning(notice)
                else:
                    st.info(notice)
        if canceling_records:
            for metadata, snapshot in canceling_records:
                cancel_info, delete_info = st.columns(
                    [0.82, 0.18], vertical_alignment="center"
                )
                with cancel_info:
                    label = JOB_KIND_LABELS.get(str(metadata.get("kind")), "后台任务")
                    st.info(
                        f"{label}已请求终止 · {job_timing_text(snapshot)}。"
                        "仍在等待当前调用释放资源，迟到结果不会保存。"
                    )
                with delete_info:
                    if st.button(
                        "删除记录",
                        key=f"forget_canceling_job_{snapshot.job_id}",
                        type="tertiary",
                        help="立即从界面删除；后台若返回，结果仍会被丢弃。",
                    ):
                        forget_background_jobs([(metadata, snapshot)])
                        st.rerun()


def terminated_tasks_sidebar() -> None:
    records = list(st.session_state.terminated_jobs)
    if not records:
        return
    with st.expander(f"已终止任务（{len(records)}）"):
        st.caption("这里只保留任务记录，未完成的模型输出不会保存。")
        for record in records:
            job_id = str(record.get("job_id", ""))
            label = JOB_KIND_LABELS.get(str(record.get("kind", "")), "后台任务")
            detail = " / ".join(
                value
                for value in (str(record.get("provider", "")), str(record.get("model", "")))
                if value
            )
            info_col, delete_col = st.columns([0.76, 0.24], vertical_alignment="center")
            with info_col:
                st.caption(f"{label} · {record.get('title', '未命名任务')}")
                if detail:
                    st.caption(detail)
                if record.get("elapsed_seconds") is not None:
                    st.caption(f"终止前运行 {format_task_duration(record['elapsed_seconds'])}")
            with delete_col:
                if st.button(
                    "删除",
                    key=f"delete_terminated_job_{job_id}",
                    type="tertiary",
                    help="删除这条已终止任务记录。",
                ):
                    st.session_state.terminated_jobs = [
                        item
                        for item in st.session_state.terminated_jobs
                        if str(item.get("job_id", "")) != job_id
                    ]
                    st.rerun()


def history_sidebar(container=None, disabled: bool = False) -> None:
    target = container if container is not None else st.sidebar
    with target:
        st.divider()
        st.subheader("内容历史")
        show_archived = st.toggle(
            "显示已归档",
            key="show_archived_history",
            disabled=disabled,
        )
        deleted_items = list_deleted_history()
        show_deleted = st.toggle(
            f"已删除（{len(deleted_items)}）",
            key="show_deleted_history",
            disabled=disabled,
        )
        if show_deleted:
            st.caption("已删除内容保留15天。点击任意条目可以恢复聊天、总结和对应的本地文件。")
            if not deleted_items:
                st.caption("目前没有可恢复的内容。")
                return
            st.markdown(
                """
                <style>
                section[data-testid="stSidebar"] .st-key-deleted_history_list button {
                    min-height: 2rem;
                    padding: 0.25rem 0.45rem;
                    border: 0;
                    border-radius: 0.5rem;
                    justify-content: flex-start;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            with st.container(key="deleted_history_list"):
                for item in deleted_items:
                    item_id = str(item["id"])
                    if st.button(
                        compact_history_title(item),
                        key=f"restore_history_{item_id}",
                        type="tertiary",
                        help=str(item.get("title") or item.get("video_title") or "未命名"),
                        use_container_width=True,
                        disabled=disabled,
                    ):
                        open_restore_dialog(item_id)
                        st.rerun()
                    st.caption(f"⏳ {deleted_history_countdown(item)}")
            return
        running_conversations_sidebar()
        terminated_tasks_sidebar()
        items = list_history(include_archived=show_archived)
        if not items:
            st.caption("生成详细总结或完成一次视频问答后，会自动保存在这里。")
            return
        item_by_id = {str(item["id"]): item for item in items}
        selected_id = str(st.session_state.selected_history_id)
        if selected_id not in item_by_id:
            opened_id = str(st.session_state.opened_history_id)
            selected_id = opened_id if opened_id in item_by_id else ""
            st.session_state.selected_history_id = selected_id

        st.markdown(
            f"""
            <style>
            section[data-testid="stSidebar"] .st-key-history_list button {{
                min-height: 2rem;
                padding: 0.25rem 0.45rem;
                border: 0;
                border-radius: 0.5rem;
                justify-content: flex-start;
            }}
            section[data-testid="stSidebar"] .st-key-history_list button p {{
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                text-align: left;
            }}
            section[data-testid="stSidebar"] .st-key-history_item_{selected_id} button {{
                background: rgba(128, 128, 128, 0.18);
                font-weight: 600;
            }}
            section[data-testid="stSidebar"] .st-key-history_list [class*="st-key-history_more_"] button {{
                justify-content: center;
                padding-left: 0.2rem;
                padding-right: 0.2rem;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

        visible_ids = list(item_by_id)[:40]
        bulk_history_manager(item_by_id, visible_ids, disabled)
        with st.container(key="history_list"):
            for item_id in visible_ids:
                item = item_by_id[item_id]
                full_title = str(item.get("title") or item.get("video_title") or "未命名")
                hover_title = re.sub(r"\s+", " ", full_title).strip()
                css_hover_title = hover_title.replace("\\", "\\\\").replace('"', '\\"')
                st.markdown(
                    f"""
                    <style>
                    section[data-testid="stSidebar"] .st-key-history_item_{item_id} {{
                        position: relative;
                    }}
                    section[data-testid="stSidebar"] .st-key-history_item_{item_id}:hover::after {{
                        content: "{css_hover_title}";
                        position: absolute;
                        left: 0;
                        bottom: calc(100% + 0.2rem);
                        z-index: 10000;
                        width: max-content;
                        max-width: min(42rem, 80vw);
                        padding: 0.38rem 0.55rem;
                        border: 1px solid rgba(128, 128, 128, 0.25);
                        border-radius: 0.45rem;
                        background: var(--secondary-background-color);
                        color: var(--text-color);
                        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.14);
                        white-space: normal;
                        pointer-events: none;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                title_col, menu_col = st.columns([0.84, 0.16], gap="small")
                with title_col:
                    if st.button(
                        compact_history_title(item),
                        key=f"history_item_{item_id}",
                        type="tertiary",
                        use_container_width=True,
                        disabled=disabled,
                    ):
                        st.session_state.selected_history_id = item_id
                        if open_history_item(item):
                            st.rerun()
                with menu_col:
                    with st.popover(
                        "⋯",
                        key=f"history_more_{item_id}",
                        type="tertiary",
                        disabled=disabled,
                        use_container_width=True,
                    ):
                        st.markdown(f"**{full_title}**")
                        kind_label = (
                            "视频对话"
                            if item.get("kind") == "video"
                            else "详细笔记"
                            if item.get("kind") == "summary"
                            else "视频问答"
                        )
                        created = str(item.get("created_at", ""))[:10]
                        elapsed = item.get("processing_seconds")
                        elapsed_text = (
                            f" · 用时 {format_task_duration(float(elapsed))}"
                            if elapsed is not None
                            else ""
                        )
                        st.caption(
                            f"{kind_label} · {created} · "
                            f"{item.get('provider', '未知服务')} / {item.get('model', '未知模型')}"
                            f"{elapsed_text}"
                        )
                        archive_label = "取消归档" if item.get("archived") else "归档"
                        if st.button(
                            archive_label,
                            key=f"archive_history_{item_id}",
                            icon=":material/archive:",
                            use_container_width=True,
                            disabled=disabled,
                        ):
                            update_history_item(
                                item_id,
                                archived=not bool(item.get("archived")),
                            )
                            st.session_state.pending_bulk_history_ids = []
                            st.rerun()
                        confirm_delete = st.checkbox(
                            "确认移至已删除",
                            key=f"confirm_delete_{item_id}",
                            disabled=disabled,
                        )
                        st.caption("聊天内容和关联本地文件保留15天，可从“已删除”恢复。")
                        if st.button(
                            "移至已删除",
                            key=f"delete_history_{item_id}",
                            icon=":material/delete:",
                            disabled=disabled or not confirm_delete,
                            use_container_width=True,
                        ):
                            try:
                                deleted = soft_delete_history_item(item_id)
                            except OSError as exc:
                                deleted = None
                                st.session_state.flash_message = (
                                    f"无法建立可恢复备份，记录尚未删除：{exc}"
                                )
                            if deleted is not None:
                                clear_deleted_history_item(item_id, item)
                                st.session_state.selected_history_id = ""
                                st.session_state.pending_bulk_history_ids = []
                                if deleted.get("content_cleanup_pending"):
                                    st.session_state.flash_message = (
                                        "记录已移入15天回收站；正在占用的文件会在任务释放后自动清理。"
                                    )
                            st.rerun()

        if len(items) > 40:
            st.caption(f"这里只显示最近 40 条；另有 {len(items) - 40} 条可通过归档整理后查看。")


@st.cache_data(ttl=10, show_spinner=False)
def ollama_models(base_url: str) -> list[str]:
    host = base_url.rstrip("/")
    if host.endswith("/v1"):
        host = host[:-3]
    try:
        response = requests.get(f"{host}/api/tags", timeout=3)
        response.raise_for_status()
        entries = [
            item
            for item in response.json().get("models", [])
            if isinstance(item, dict) and item.get("name")
        ]

        models: list[str] = []
        seen_digests: set[str] = set()
        for item in sorted(
            entries,
            key=lambda entry: (
                str(entry["name"]).lower().startswith("hf.co/"),
                str(entry["name"]).lower(),
            ),
        ):
            digest = str(item.get("digest") or item.get("model") or item["name"])
            if digest in seen_digests:
                continue
            seen_digests.add(digest)
            models.append(str(item["name"]))
        return models
    except (requests.RequestException, ValueError, TypeError):
        return []


@st.cache_data(ttl=60, show_spinner=False)
def ollama_model_context_limit(base_url: str, model: str) -> int | None:
    host = base_url.rstrip("/")
    if host.endswith("/v1"):
        host = host[:-3]
    try:
        response = requests.post(f"{host}/api/show", json={"model": model}, timeout=10)
        response.raise_for_status()
        model_info = response.json().get("model_info", {})
        limits = [
            int(value)
            for key, value in model_info.items()
            if key.endswith(".context_length") and isinstance(value, (int, float))
        ]
        return max(limits) if limits else None
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        return None


def rounded_context_window(required_tokens: int) -> int:
    value = 2_048
    while value < required_tokens:
        value *= 2
    return value


def _find_context_limit(value: object) -> int | None:
    context_keys = {
        "context_length",
        "max_context_length",
        "context_window",
        "max_model_len",
        "max_input_tokens",
        "input_token_limit",
        "n_ctx_train",
    }
    found: list[int] = []

    def visit(item: object) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if str(key).lower() in context_keys:
                    try:
                        found.append(int(child))
                    except (TypeError, ValueError):
                        visit(child)
                else:
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return max(found) if found else None


@st.cache_data(ttl=30, show_spinner=False)
def openai_compatible_catalog(base_url: str, api_key: str) -> list[dict[str, object]]:
    base = base_url.rstrip("/")
    if not base:
        return []
    endpoints = [f"{base}/models"]
    root = base[:-3] if base.endswith("/v1") else base
    native_endpoint = f"{root}/api/v1/models"
    if native_endpoint not in endpoints:
        endpoints.append(native_endpoint)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    catalog: dict[str, int | None] = {}
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, headers=headers, timeout=3)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                entries = payload
            elif isinstance(payload, dict):
                entries = payload.get("data") or payload.get("models") or []
            else:
                entries = []
            if isinstance(entries, dict):
                entries = list(entries.values())
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                model_id = next(
                    (
                        str(entry[key])
                        for key in ("id", "model", "key", "identifier", "model_key")
                        if entry.get(key)
                    ),
                    "",
                )
                if not model_id:
                    continue
                detected = _find_context_limit(entry)
                existing = catalog.get(model_id)
                catalog[model_id] = max(existing or 0, detected or 0) or None
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            continue
    return [
        {"id": model_id, "context_window": context}
        for model_id, context in sorted(catalog.items(), key=lambda item: item[0].lower())
    ]


def context_window_panel(
    provider: str,
    model: str,
    max_context_window: int | None,
    default_context_window: int,
) -> int:
    transcript = st.session_state.get("transcript")
    required_context = estimate_context_tokens(transcript) if transcript else None
    recommended_context = rounded_context_window(required_context or default_context_window)
    if max_context_window:
        recommended_context = min(recommended_context, max_context_window)
    context_key = f"context_window_{provider}_{model}"
    minimum_context = min(2_048, max_context_window) if max_context_window else 2_048
    if context_key not in st.session_state:
        st.session_state[context_key] = max(
            minimum_context,
            min(default_context_window, max_context_window or default_context_window),
        )
    current_context = int(st.session_state[context_key])
    if required_context and current_context < required_context:
        if max_context_window is None or required_context <= max_context_window:
            st.warning(
                f"完整字幕估计需要约 {required_context:,} tokens；当前预算为 {current_context:,}。"
                "提高上下文可以减少字幕筛选造成的信息遗漏。"
            )
            if st.button(
                f"使用建议值 {recommended_context:,}",
                key=f"apply_context_{provider}_{model}",
                use_container_width=True,
            ):
                st.session_state[context_key] = recommended_context
        else:
            st.warning(
                f"完整字幕估计需要约 {required_context:,} tokens，但检测到的模型上限为 "
                f"{max_context_window:,}。程序会检索相关字幕；整部视频类问题可能遗漏内容，"
                "建议改用上下文更长的模型。"
            )
    context_window = int(
        st.number_input(
            "上下文预算（tokens）",
            min_value=minimum_context,
            max_value=max_context_window,
            step=1_024 if minimum_context >= 2_048 else 256,
            key=context_key,
            help="它决定程序最多发送多少字幕和对话。数值越大通常越完整，但云端成本可能提高，"
            "本地服务的内存或显存占用也可能增加。",
        )
    )
    if max_context_window:
        st.caption(f"检测到的模型最大上下文：{max_context_window:,} tokens。")
        if max_context_window < 2_048:
            st.warning("该模型上下文小于 2,048 tokens，长字幕总结和问答可能无法可靠运行。")
    else:
        st.caption("未检测到模型上限，请按照模型或服务文档填写；超限时程序会显示接口错误。")
    if provider == "OpenAI 兼容（自定义）":
        st.caption(
            "标准 OpenAI 接口不能替服务端扩大上下文；此数值应与 LM Studio、LocalAI、"
            "llama.cpp、vLLM 或云端服务中的实际设置一致。"
        )
    if provider == "Ollama":
        st.info(
            "这里设置的是程序允许使用的上下文上限，不会强制每次都分配这么多。"
            "程序会按本次字幕、问题和输出长度自动选择实际运行窗口，减少显存/内存浪费。"
        )
        if context_window >= 65_536:
            st.warning(
                "你选择了很大的本地上下文上限。处理真正接近该长度的内容时仍可能非常慢，"
                "并需要大量内存或显存；普通电脑建议先使用 8,192～32,768。"
            )
    st.caption(
        "总结和问答的输出长度会根据上下文自动调整；只要模型仍因长度上限停止，程序就会持续续写，"
        "直到模型正常结束。云端模型的多次续写会增加 API 用量和费用。"
    )
    return context_window


def llm_settings_panel() -> LLMSettings:
    st.subheader("AI 设置")
    saved_settings = st.session_state.llm_settings
    provider = st.selectbox(
        "模型服务",
        ["Gemini", "DeepSeek", "OpenAI", "Anthropic", "OpenAI 兼容（自定义）", "Ollama"],
        key="model_provider",
    )
    defaults = {
        "Gemini": ("gemini-2.5-flash", ""),
        "DeepSeek": ("deepseek-v4-flash", "https://api.deepseek.com"),
        "OpenAI": ("gpt-4.1-mini", "https://api.openai.com/v1"),
        "Anthropic": ("claude-sonnet-5", ""),
        "OpenAI 兼容（自定义）": ("", "http://localhost:1234/v1"),
        "Ollama": ("", "http://localhost:11434/v1"),
    }
    known_limits = {
        "Gemini": 1_048_576,
        "DeepSeek": 1_000_000,
        "OpenAI": 1_047_576,
        "Anthropic": 1_000_000,
    }
    default_model, default_url = defaults[provider]
    saved_for_provider = saved_settings.provider == provider
    if provider == "Ollama":
        api_key = ""
        base_url = st.text_input(
            "接口地址",
            value=saved_settings.base_url if saved_for_provider else default_url,
            key="base_url_Ollama",
            help="默认连接本机 Ollama；如果服务在另一台设备上，请填写那台设备可访问的地址。",
        )
        installed_models = ollama_models(base_url)
        max_context_window = None
        if installed_models:
            selected_index = (
                installed_models.index(saved_settings.model)
                if saved_for_provider and saved_settings.model in installed_models
                else 0
            )
            model = st.selectbox(
                "可用模型",
                installed_models,
                index=selected_index,
                key="ollama_installed_model",
            )
            st.caption(f"已连接 Ollama，检测到 {len(installed_models)} 个可用模型，无需 API Key。")
            max_context_window = ollama_model_context_limit(base_url, model)
        else:
            model = st.text_input(
                "模型名称",
                value=saved_settings.model if saved_for_provider else default_model,
                key="model_Ollama",
                placeholder="请输入 Ollama 中已经安装的模型名称",
            )
            st.warning("无法读取 Ollama 模型列表。请确认服务已启动、接口地址正确，并且至少安装了一个模型。")
        st.info(
            "Ollama 模式只识别当前接口服务中已经安装的模型。请先运行 `ollama pull 模型名`；"
            "单独下载的 `.gguf` 或 `.safetensors` 文件不会被本程序直接打开。"
        )
        context_window = context_window_panel(provider, model, max_context_window, 8_192)
    elif provider == "OpenAI 兼容（自定义）":
        base_url = st.text_input(
            "接口地址",
            value=saved_settings.base_url if saved_for_provider else default_url,
            key="custom_base_url",
        )
        api_key = st.text_input(
            "API Key（本地服务通常可留空）",
            value=saved_settings.api_key if saved_for_provider else "",
            type="password",
            key="custom_api_key",
        )
        catalog = openai_compatible_catalog(base_url, api_key)
        if catalog:
            model_ids = [str(item["id"]) for item in catalog]
            selected_index = (
                model_ids.index(saved_settings.model)
                if saved_for_provider and saved_settings.model in model_ids
                else 0
            )
            model = st.selectbox(
                "服务中的可用模型",
                model_ids,
                index=selected_index,
                key=f"custom_model_{base_url}",
            )
            selected = next(item for item in catalog if item["id"] == model)
            max_context_window = selected.get("context_window")
            max_context_window = int(max_context_window) if max_context_window else None
            st.caption(f"已通过 `/models` 检测到 {len(model_ids)} 个模型。")
        else:
            model = st.text_input(
                "模型 ID",
                value=saved_settings.model if saved_for_provider else "",
                key="custom_model_manual",
                placeholder="填写服务端显示的模型 ID",
            )
            max_context_window = None
            st.warning(
                "没有读取到模型列表。请确认服务已启动、地址包含正确的 `/v1` 路径；"
                "也可以手动填写模型 ID。"
            )
        st.info(
            "适用于提供 OpenAI Chat Completions 接口的本地或云端服务，例如 LM Studio、"
            "LocalAI、llama.cpp、vLLM 和多数 API 聚合平台。不同服务的地址和 Key 要求请看其文档。"
        )
        st.caption(
            "使用提示：先在对应软件中加载并启动模型服务，再填写兼容接口的 `/v1` 地址；"
            "本地服务通常可留空 Key，云端服务通常必须填写。最后点击“测试 AI 连接”。"
        )
        context_window = context_window_panel(provider, model, max_context_window, 8_192)
    else:
        model = st.text_input(
            "模型名称",
            value=saved_settings.model if saved_for_provider else default_model,
            key=f"model_{provider}",
        )
        api_key = st.text_input(
            "API Key",
            value=saved_settings.api_key if saved_for_provider else "",
            type="password",
            key=f"api_key_{provider}",
        )
        if provider in ("Gemini", "Anthropic"):
            base_url = ""
        else:
            base_url = st.text_input(
                "接口地址",
                value=saved_settings.base_url if saved_for_provider else default_url,
                key=f"base_url_{provider}",
            )
        max_context_window = known_limits.get(provider) if model.strip() == default_model else None
        context_window = context_window_panel(provider, model, max_context_window, 32_768)
        provider_hints = {
            "Gemini": "填写 Google AI Studio/Google Cloud 可用的 Gemini API Key；网页端 Gemini 会员不能代替 API Key。",
            "DeepSeek": "填写 DeepSeek 开放平台的 API Key；DeepSeek 网页或 App 账号额度不能代替 API 额度。",
            "OpenAI": "填写 OpenAI API 平台的 API Key；ChatGPT 订阅不能代替 API 额度。",
            "Anthropic": "填写 Anthropic Console 的 API Key；Claude 网页订阅不能代替 API 额度。",
        }
        st.info(provider_hints[provider])
        st.caption(
            "如果修改了默认模型名称且程序无法检测上限，请根据服务商文档调整上下文预算。"
        )
    return LLMSettings(
        provider=provider,
        model=model.strip(),
        api_key=api_key.strip(),
        base_url=base_url.strip(),
        context_window=context_window,
        max_context_window=max_context_window,
    )


def reader_settings_panel() -> dict[str, str]:
    st.subheader("本地读取设置")
    auth_mode = st.selectbox(
        "B站访问身份（普通公开视频无需登录）",
        ["不使用", "读取浏览器", "cookies.txt 文件"],
        key="reader_auth_mode",
        format_func=lambda value: {
            "不使用": "不使用登录信息（推荐）",
            "读取浏览器": "读取本机浏览器中的 B站登录信息",
            "cookies.txt 文件": "使用导出的 cookies.txt 登录信息",
        }[value],
        help="它不是让你在本程序里登录，而是让 yt-dlp 在必要时借用你已有的 B站登录状态。",
    )
    st.caption(
        "绝大多数公开视频保持“不使用”即可。只有登录后才能播放、仅会员可见或 B站要求验证身份时，"
        "才尝试另外两项；这不能绕过付费、地区、私密或平台权限限制。"
    )
    browser_label = st.session_state.reader_settings.get("browser", "Edge")
    cookie_path = st.session_state.reader_settings.get("cookie_path", "")
    if auth_mode == "读取浏览器":
        browser_label = st.selectbox(
            "浏览器",
            ["Edge", "Chrome", "Firefox"],
            key="reader_browser",
        )
        st.caption("浏览器运行时可能锁定 Cookie 数据库；读取失败时请完全退出所选浏览器后重试。")
    elif auth_mode == "cookies.txt 文件":
        cookie_path = st.text_input(
            "cookies.txt 的本机绝对路径",
            value=cookie_path,
            key="reader_cookie_path",
        )
    whisper_model = st.selectbox(
        "Whisper 模型",
        ["tiny", "base", "small", "medium", "large-v3"],
        key="reader_whisper_model",
        help="模型越大通常越准确，但需要更多内存或显存且处理更慢。",
    )
    st.caption("Whisper 是把视频里的说话声转换成文字；small 兼顾准确度和普通电脑速度。")
    whisper_device = st.selectbox(
        "Whisper 运行设备",
        ["CPU（兼容性优先）", "自动检测 GPU"],
        key="reader_whisper_device",
        help="GPU 模式需要受支持的 NVIDIA 显卡及正确配置的 CUDA 运行环境。",
    )
    st.caption("不确定时选 CPU；GPU 更快，但仅在兼容的显卡与 CUDA 环境配置正确时有效。")
    whisper_language = st.selectbox(
        "语音语言",
        ["中文", "自动检测", "英文"],
        key="reader_whisper_language",
    )
    st.caption("已知主要语言时直接选择可提高识别稳定性；混合语言视频建议使用自动检测。")
    visual_fallback_sensitivity = st.selectbox(
        "文字不足时补充画面的敏感度",
        ["节省费用", "标准（推荐）", "严格完整"],
        key="reader_visual_fallback_sensitivity",
        help="越严格越容易在字幕或转写偏少时补充画面，但处理更慢，云端视觉模型费用也可能更高。",
    )
    st.caption(
        "标准档约要求30个有效文字单位/分钟，并覆盖至少35%的视频时间线；"
        "节省档减少画面调用，严格档更重视内容覆盖。"
    )
    with st.expander("这些设置分别在什么时候使用？"):
        st.markdown(
            "- **B站访问身份**：只影响读取视频、字幕和下载媒体，不会登录 AI。\n"
            "- **Whisper 模型**：只有没有字幕或主动选择重新转写时才使用。\n"
            "- **Whisper 运行设备**：决定语音转写使用 CPU 还是尝试 GPU。\n"
            "- **语音语言**：告诉 Whisper 应重点识别哪种语言。\n"
            "- **画面补充敏感度**：根据有效文字密度和时间线覆盖决定是否继续分析截图。\n"
            "- **画面读取**：不使用 Whisper，而是抽取截图交给当前支持图片输入的 AI 模型。"
        )
    st.info(
        "Cookie 只由 yt-dlp 用于向 B站发起请求，不会发送给 AI 服务，也不会由本程序保存。"
        "选择云端 AI 时，字幕文本会发送给相应服务；需要完全本地处理请选择 Ollama。"
    )
    return {
        "auth_mode": auth_mode,
        "browser": browser_label,
        "cookie_path": cookie_path,
        "whisper_model": whisper_model,
        "whisper_device": whisper_device,
        "whisper_language": whisper_language,
        "visual_fallback_sensitivity": visual_fallback_sensitivity,
    }


def prepare_settings_widgets() -> None:
    settings = st.session_state.llm_settings
    reader = st.session_state.reader_settings
    widget_defaults = {
        "model_provider": settings.provider,
        f"context_window_{settings.provider}_{settings.model}": settings.context_window,
        "reader_auth_mode": reader.get("auth_mode", "不使用"),
        "reader_browser": reader.get("browser", "Edge"),
        "reader_cookie_path": reader.get("cookie_path", ""),
        "reader_whisper_model": reader.get("whisper_model", "small"),
        "reader_whisper_device": reader.get("whisper_device", "CPU（兼容性优先）"),
        "reader_whisper_language": reader.get("whisper_language", "中文"),
        "reader_visual_fallback_sensitivity": reader.get(
            "visual_fallback_sensitivity", "标准（推荐）"
        ),
    }
    for key, value in widget_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def validate_llm_settings(settings: LLMSettings) -> str | None:
    if not settings.model.strip():
        return "尚未选择模型。请在“AI 设置”中选择或填写模型名称。"
    if settings.context_window < 2_048:
        return "上下文预算不能小于 2,048 tokens。请在“AI 设置”中调高该数值。"
    if settings.provider in {"Gemini", "DeepSeek", "OpenAI", "Anthropic"}:
        if not settings.api_key.strip():
            return f"尚未填写 {settings.provider} API Key。请在“AI 设置”中填写后重试。"
    if settings.provider in {"Ollama", "OpenAI 兼容（自定义）"}:
        if not settings.base_url.strip():
            return "接口地址为空。请在“AI 设置”中填写模型服务的接口地址。"
    if settings.provider == "Ollama":
        host = settings.base_url.rstrip("/")
        if host.endswith("/v1"):
            host = host[:-3]
        try:
            response = requests.get(f"{host}/api/tags", timeout=3)
            response.raise_for_status()
            installed = {
                str(item.get("name", ""))
                for item in response.json().get("models", [])
                if isinstance(item, dict)
            }
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            return "无法连接 Ollama。请在“AI 设置”中检查接口地址，并确认 Ollama 已经启动。"
        if settings.model not in installed:
            return (
                f"Ollama 中没有模型“{settings.model}”。请在“AI 设置”中选择当前服务已经安装的模型。"
            )
    return None


def friendly_llm_error(settings: LLMSettings, exc: Exception) -> tuple[str, bool]:
    detail = str(exc).strip() or exc.__class__.__name__
    lowered = detail.lower()
    if any(marker in lowered for marker in ("api key", "api_key", "unauthorized", "401", "authentication")):
        return (f"{settings.provider} API Key 缺失、无效或无权限。请打开设置检查 API Key。\n\n原始错误：{detail}", True)
    if any(marker in lowered for marker in ("model not found", "model_not_found", "没有模型", "404")):
        return (f"当前服务找不到模型“{settings.model}”。请打开设置重新选择模型。\n\n原始错误：{detail}", True)
    if any(marker in lowered for marker in ("context", "num_ctx", "上下文", "token limit")):
        return (f"当前上下文设置超出模型或硬件可承受范围。请打开设置减小上下文预算。\n\n原始错误：{detail}", True)
    if any(marker in lowered for marker in ("connection", "connect", "无法连接", "refused", "接口地址")):
        return (f"无法连接当前模型服务。请打开设置检查服务是否启动以及接口地址。\n\n原始错误：{detail}", True)
    if "timeout" in lowered or "超时" in detail:
        return (f"模型长时间没有返回数据。可在设置中减小上下文预算或换用更小的本地模型。\n\n原始错误：{detail}", True)
    return (f"模型运行失败：{detail}", False)


def open_settings_with_error(message: str) -> None:
    prepare_settings_widgets()
    st.session_state.settings_error_message = message
    st.session_state.settings_dialog_open = True


@st.dialog("设置")
def settings_dialog() -> None:
    st.caption("模型与读取参数只在需要时展开；关闭后侧栏继续优先显示内容历史。")
    if st.session_state.settings_error_message:
        st.error(st.session_state.settings_error_message)
    settings = llm_settings_panel()
    locking_jobs = current_model_jobs(lock_only=True)
    releasing_jobs = [
        record for record in current_model_jobs() if record[1].status == "canceling"
    ]
    candidate_identity = (settings.provider, settings.model, settings.base_url)
    locked_identities = {
        (locked.provider, locked.model, locked.base_url)
        for metadata, _ in locking_jobs
        if isinstance((locked := metadata.get("settings")), LLMSettings)
    }
    model_changed = bool(
        locked_identities
        and any(candidate_identity != locked_identity for locked_identity in locked_identities)
    )
    model_change_locked = bool(locking_jobs and model_changed)
    if locking_jobs:
        locked_models = sorted(
            {
                f"{metadata['settings'].provider} / {metadata['settings'].model}"
                for metadata, _ in locking_jobs
                if isinstance(metadata.get("settings"), LLMSettings)
            }
        )
        st.warning(
            "当前对话仍有模型任务运行，任务已锁定启动时的模型："
            + ("、".join(locked_models) or "当前模型")
            + "。可以查看其他设置，但不能直接保存为另一模型。"
        )
        if st.button(
            "终止当前对话后台任务并允许切换模型",
            key="cancel_jobs_before_model_switch",
            type="secondary",
            use_container_width=True,
        ):
            canceled = cancel_current_workspace_jobs(model_only=True)
            st.session_state.settings_error_message = (
                f"已向 {canceled} 个任务发出终止请求。若模型正在处理一次请求，资源可能稍后才完全释放。"
            )
            st.session_state.settings_dialog_open = True
            st.rerun()
    elif releasing_jobs:
        st.info("原模型任务正在释放资源；现在可以保存另一模型，但立即启动新任务可能暂时变慢。")
    remember_api_key = st.checkbox(
        "在这台电脑上保存 API Key",
        key="remember_api_key",
        help="Key 会保存在本机 data/settings.json（该目录不会提交到 GitHub）。共享电脑请关闭。",
    )
    if remember_api_key and settings.api_key:
        st.warning("API Key 将以本机配置文件形式保存。请勿分享 data/settings.json，也不要在公共电脑上启用。")
    if st.button(
        "测试 AI 连接",
        use_container_width=True,
        key="dialog_test_ai",
        disabled=model_change_locked,
        help="当前对话运行期间如需测试另一模型，请先终止该对话的后台任务。" if model_change_locked else None,
    ):
        try:
            with st.spinner("正在测试模型和接口……"):
                test_reply = test_connection(settings)
            st.success(f"连接成功：{test_reply[:80]}")
        except Exception as exc:
            st.error(f"连接失败：{exc}")
    st.divider()
    reader_settings = reader_settings_panel()
    if model_change_locked:
        st.error("模型切换尚未保存。请保持原模型，或先终止当前对话的后台任务。")
    if st.button(
        "保存并关闭",
        type="primary",
        use_container_width=True,
        key="close_settings",
        disabled=model_change_locked,
    ):
        st.session_state.llm_settings = settings
        st.session_state.reader_settings = reader_settings
        save_app_settings(
            {
                "remember_api_key": remember_api_key,
                "llm": {
                    "provider": settings.provider,
                    "model": settings.model,
                    "api_key": settings.api_key if remember_api_key else "",
                    "base_url": settings.base_url,
                    "context_window": settings.context_window,
                    "max_context_window": settings.max_context_window,
                },
                "reader": dict(reader_settings),
            }
        )
        st.session_state.settings_error_message = ""
        st.session_state.ai_error = None
        st.session_state.settings_dialog_open = False
        st.session_state.flash_message = "设置已保存在这台电脑上，下次启动会自动恢复。"
        st.rerun()


def schedule_ai_job(kind: str, transcript: Transcript, settings: LLMSettings, **payload) -> None:
    issue = validate_llm_settings(settings)
    if issue:
        open_settings_with_error(issue)
        st.session_state.ai_error = {"scope": kind, "message": issue}
        st.rerun()
    st.session_state.settings_dialog_open = False
    st.session_state.settings_error_message = ""
    st.session_state.ai_error = None
    remember_workspace_in_url(str(st.session_state.workspace_id))
    settings_snapshot = LLMSettings(
        provider=settings.provider,
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        context_window=settings.context_window,
        max_context_window=settings.max_context_window,
    )
    history_snapshot = list(st.session_state.video_chats.get(transcript.video_id, []))
    branches_snapshot, active_branch_id = normalize_chat_branches(
        transcript.video_id, history_snapshot
    )
    if "history_override" in payload:
        history_snapshot = list(payload["history_override"])
    question = str(payload.get("question", ""))
    manager = get_job_manager()
    if kind == "summary":
        job_id = manager.submit(
            lambda progress: summarize(transcript, settings_snapshot, progress=progress),
            "正在准备详细视频笔记……",
        )
    else:
        job_id = manager.submit(
            lambda progress: answer_question(
                transcript,
                question,
                settings_snapshot,
                history_snapshot,
                progress=progress,
            ),
            "正在分析问题并检索视频内容……",
        )
    is_local = settings.provider == "Ollama" or (
        settings.provider == "OpenAI 兼容（自定义）"
        and any(host in settings.base_url.lower() for host in ("localhost", "127.0.0.1", "0.0.0.0"))
    )
    metadata = {
        "job_id": job_id,
        "job_group": "ai",
        "kind": kind,
        "video_id": transcript.video_id,
        "transcript": transcript,
        "settings": settings_snapshot,
        "question": question,
        "history": history_snapshot,
        "chat_branches": branches_snapshot,
        "active_chat_branch": active_branch_id,
        "edit_index": payload.get("edit_index"),
        "new_branch_id": str(payload.get("new_branch_id", "")),
        "history_item_id": str(st.session_state.opened_history_id),
        "local": is_local,
        "workspace_id": st.session_state.workspace_id,
    }
    manager.set_metadata(job_id, metadata)
    estimated_seconds, estimate_note = estimate_ai_job_seconds(
        kind, transcript, settings_snapshot
    )
    manager.set_estimate(job_id, estimated_seconds, estimate_note)
    st.session_state.background_ai_jobs[job_id] = metadata
    label = JOB_KIND_LABELS.get(kind, "AI任务")
    if estimated_seconds >= LONG_TASK_SECONDS:
        st.session_state.flash_message = (
            f"{label}预计约 {format_task_duration(estimated_seconds)}，可能需要较长时间；"
            "任务已在后台运行。"
        )
    else:
        st.session_state.flash_message = f"{label}已在后台开始。"
    st.rerun()


def restore_source_branch_after_interrupted_edit(metadata: dict[str, object]) -> None:
    if not isinstance(metadata.get("edit_index"), int):
        return
    transcript = metadata.get("transcript")
    if not isinstance(transcript, Transcript):
        return
    branches = [
        dict(branch)
        for branch in metadata.get("chat_branches", [])
        if isinstance(branch, dict)
    ]
    active_branch_id = str(metadata.get("active_chat_branch", ""))
    source = next(
        (
            branch
            for branch in branches
            if str(branch.get("id")) == active_branch_id
        ),
        None,
    )
    if source is None:
        return
    history = list(source.get("history", []))
    st.session_state.video_chat_branches[transcript.video_id] = branches
    st.session_state.video_active_chat_branches[transcript.video_id] = active_branch_id
    st.session_state.video_chats[transcript.video_id] = history
    if st.session_state.workspace_id == metadata.get("workspace_id"):
        st.session_state.chat_history = history


def finalize_background_ai_jobs() -> bool:
    manager = get_job_manager()
    changed = False
    finished_ids: list[str] = []
    for job_id, metadata in list(st.session_state.background_ai_jobs.items()):
        snapshot = manager.snapshot(job_id)
        if snapshot is None or snapshot.status not in {"completed", "failed", "canceled"}:
            continue
        changed = True
        finished_ids.append(job_id)
        transcript = metadata["transcript"]
        settings = metadata["settings"]
        kind = str(metadata["kind"])
        if snapshot.status == "canceled":
            restore_source_branch_after_interrupted_edit(metadata)
            remember_terminated_job(metadata, snapshot)
            if st.session_state.workspace_id == metadata.get("workspace_id"):
                st.session_state.flash_message = "后台模型任务已终止；未完成的结果没有保存。"
            manager.discard(job_id)
            continue
        if snapshot.status == "failed":
            restore_source_branch_after_interrupted_edit(metadata)
            if metadata.get("kind") == "audio" and metadata.get("part"):
                st.session_state.audio_failures[metadata["part"]["id"]] = str(
                    snapshot.error or snapshot.message
                )
            error = snapshot.error or RuntimeError(snapshot.message)
            message, is_configuration_error = friendly_llm_error(settings, error)
            st.session_state.background_ai_errors.append(
                {
                    "title": transcript.title,
                    "kind": kind,
                    "message": message,
                    "elapsed": job_elapsed_seconds(snapshot),
                }
            )
            if st.session_state.workspace_id == metadata.get("workspace_id"):
                st.session_state.ai_error = {"scope": kind, "message": message}
                if is_configuration_error:
                    st.session_state.settings_error_message = message
            manager.discard(job_id)
            continue

        result = str(snapshot.result or "").strip()
        if kind == "summary":
            history_item = save_video_history_item(
                transcript=transcript,
                provider=settings.provider,
                model=settings.model,
                summary_content=result,
                processing_seconds=job_elapsed_seconds(snapshot),
                preferred_item_id=str(metadata.get("history_item_id", "")),
            )
            st.session_state.video_summaries[transcript.video_id] = result
            if st.session_state.workspace_id == metadata.get("workspace_id"):
                st.session_state.summary = result
                st.session_state.opened_history_id = history_item["id"]
        else:
            question = str(metadata.get("question", ""))
            chat_history = list(metadata.get("history", []))
            chat_history.append({"question": question, "answer": result})
            branches = [
                dict(branch)
                for branch in metadata.get("chat_branches", [])
                if isinstance(branch, dict)
            ]
            active_branch_id = str(metadata.get("active_chat_branch", ""))
            edit_index = metadata.get("edit_index")
            new_branch_id = str(metadata.get("new_branch_id", ""))
            if isinstance(edit_index, int) and new_branch_id:
                for branch in branches:
                    if str(branch.get("id")) == active_branch_id:
                        branch["fork_index"] = edit_index
                branches.append(
                    {
                        "id": new_branch_id,
                        "history": chat_history,
                        "fork_index": edit_index,
                    }
                )
                active_branch_id = new_branch_id
            else:
                active = next(
                    (
                        branch
                        for branch in branches
                        if str(branch.get("id")) == active_branch_id
                    ),
                    None,
                )
                if active is not None:
                    active["history"] = chat_history
            history_item = save_video_history_item(
                transcript=transcript,
                provider=settings.provider,
                model=settings.model,
                chat_history=chat_history,
                chat_branches=branches,
                active_chat_branch=active_branch_id,
                processing_seconds=job_elapsed_seconds(snapshot),
                preferred_item_id=str(metadata.get("history_item_id", "")),
            )
            st.session_state.video_chat_branches[transcript.video_id] = branches
            st.session_state.video_active_chat_branches[
                transcript.video_id
            ] = active_branch_id
            if st.session_state.workspace_id == metadata.get("workspace_id"):
                st.session_state.video_chats[transcript.video_id] = chat_history
                st.session_state.chat_history = chat_history
                st.session_state.opened_history_id = history_item["id"]
        st.session_state.flash_message = (
            f"后台任务已完成并保存：{transcript.title} · "
            f"用时 {format_task_duration(job_elapsed_seconds(snapshot))}"
        )
        manager.discard(job_id)
    for job_id in finished_ids:
        st.session_state.background_ai_jobs.pop(job_id, None)
    return changed


def apply_completed_media(entry: dict[str, object], *, new_workspace: bool = True) -> None:
    if entry.get("kind") == "inspect":
        st.session_state.parts = list(entry.get("result") or [])
        st.session_state.inspected_url = str(entry.get("url", ""))
        st.session_state.transcript = None
        st.session_state.summary = ""
        st.session_state.chat_history = []
        st.session_state.current_video_id = ""
    else:
        result = entry.get("result")
        if isinstance(result, SmartReadResult):
            activate_transcript(result.transcript)
            if result.used_visual:
                notice, is_local = visual_accuracy_notice(entry.get("settings"))
                st.session_state.visual_accuracy_notices[result.transcript.video_id] = {
                    "message": notice,
                    "local": is_local,
                }
            else:
                st.session_state.visual_accuracy_notices.pop(
                    result.transcript.video_id, None
                )
            if entry.get("kind") == "reread":
                st.session_state.parts = []
    if new_workspace:
        st.session_state.workspace_id = uuid4().hex
        remember_workspace_in_url(st.session_state.workspace_id)


def finalize_background_media_jobs() -> bool:
    manager = get_job_manager()
    changed = False
    finished_ids: list[str] = []
    for job_id, metadata in list(st.session_state.background_media_jobs.items()):
        snapshot = manager.snapshot(job_id)
        if snapshot is None or snapshot.status not in {"completed", "failed", "canceled"}:
            continue
        changed = True
        finished_ids.append(job_id)
        if snapshot.status == "canceled":
            remember_terminated_job(metadata, snapshot)
            if st.session_state.workspace_id == metadata.get("workspace_id"):
                st.session_state.flash_message = "后台视频读取任务已终止；未完成的结果没有保存。"
            manager.discard(job_id)
            continue
        if snapshot.status == "failed":
            st.session_state.background_media_errors.append(
                {
                    "title": str(metadata.get("part", {}).get("title", metadata.get("url", "视频任务"))),
                    "message": str(snapshot.error or snapshot.message),
                    "elapsed": job_elapsed_seconds(snapshot),
                }
            )
            manager.discard(job_id)
            continue

        entry = {**metadata, "result": snapshot.result}
        entry["elapsed_seconds"] = job_elapsed_seconds(snapshot)
        if metadata.get("kind") != "inspect" and isinstance(snapshot.result, SmartReadResult):
            save_transcript(snapshot.result.transcript)
            if metadata.get("kind") in {"audio", "smart", "reread"} and metadata.get("part"):
                st.session_state.audio_failures.pop(metadata["part"]["id"], None)
            entry["title"] = snapshot.result.transcript.title
            entry["message"] = snapshot.result.message
            entry["warnings"] = snapshot.result.warnings
        else:
            entry["title"] = str(metadata.get("url", "视频信息"))
            entry["message"] = "视频信息、字幕和音轨检测完成。"
            entry["warnings"] = []
        st.session_state.background_media_completed.insert(0, entry)
        st.session_state.background_media_completed = st.session_state.background_media_completed[:8]
        if st.session_state.workspace_id == metadata.get("workspace_id"):
            apply_completed_media(entry, new_workspace=False)
            st.session_state.flash_message = (
                f"{entry['message']} · "
                f"用时 {format_task_duration(job_elapsed_seconds(snapshot))}"
            )
        manager.discard(job_id)
    for job_id in finished_ids:
        st.session_state.background_media_jobs.pop(job_id, None)
    return changed


@st.fragment(run_every="1s")
def background_ai_panel() -> None:
    ai_changed = finalize_background_ai_jobs()
    media_changed = finalize_background_media_jobs()
    changed = ai_changed or media_changed
    manager = get_job_manager()
    active: list[tuple[dict[str, object], object]] = []
    for job_id, metadata in st.session_state.background_ai_jobs.items():
        snapshot = manager.snapshot(job_id)
        if snapshot is not None:
            active.append((metadata, snapshot))
    for job_id, metadata in st.session_state.background_media_jobs.items():
        snapshot = manager.snapshot(job_id)
        if snapshot is not None:
            active.append((metadata, snapshot))
    if active:
        active = [record for record in active if record[1].status in ACTIVE_JOB_STATUSES]
        running = [record for record in active if record[1].status in RUNNING_JOB_STATUSES]
        releasing = [record for record in active if record[1].status == "canceling"]
        if running:
            st.caption(f"后台运行：{len(running)} 个任务")
        if releasing:
            st.caption(f"正在终止并释放资源：{len(releasing)} 个任务")
        local_count = sum(1 for metadata, _ in running if metadata.get("local"))
        releasing_local_count = sum(
            1 for metadata, _ in releasing if metadata.get("local")
        )
        if local_count >= 2:
            st.warning(
                f"正在同时运行 {local_count} 个本地模型任务。它们会争抢显存、内存和算力，"
                "生成速度可能明显下降，显存不足时还可能失败。"
            )
        if releasing_local_count:
            st.caption(
                f"另有 {releasing_local_count} 个本地任务已停止接收结果，"
                "正在等待当前模型调用返回并释放显存/内存。"
            )
    if st.session_state.background_ai_errors:
        with st.expander(f"后台失败任务（{len(st.session_state.background_ai_errors)}）"):
            for error in st.session_state.background_ai_errors[-5:]:
                elapsed = format_task_duration(float(error.get("elapsed", 0)))
                st.error(f"{error['title']}（运行 {elapsed}）：{error['message']}")
            if st.button("清除失败提示", key="clear_background_errors"):
                st.session_state.background_ai_errors = []
                st.rerun(scope="fragment")
    if st.session_state.background_media_errors:
        with st.expander(f"视频读取失败（{len(st.session_state.background_media_errors)}）"):
            for error in st.session_state.background_media_errors[-5:]:
                elapsed = format_task_duration(float(error.get("elapsed", 0)))
                st.error(f"{error['title']}（运行 {elapsed}）：{error['message']}")
            if st.button("清除读取错误", key="clear_background_media_errors"):
                st.session_state.background_media_errors = []
                st.rerun(scope="fragment")
    if changed:
        st.rerun()


def show_ai_error(scope: str) -> None:
    error = st.session_state.ai_error
    if not isinstance(error, dict) or error.get("scope") != scope:
        return
    st.error(str(error.get("message", "模型运行失败。")))
    open_col, dismiss_col = st.columns(2)
    if open_col.button(
        "打开设置修改",
        key=f"open_settings_for_{scope}",
        use_container_width=True,
    ):
        open_settings_with_error(str(error.get("message", "请检查 AI 设置。")))
        st.rerun()
    if dismiss_col.button(
        "关闭提示",
        key=f"dismiss_ai_error_{scope}",
        use_container_width=True,
    ):
        st.session_state.ai_error = None
        st.rerun()


@st.fragment(run_every="1s")
def pending_video_qa_messages(video_id: str, workspace_id: str) -> None:
    """Refresh only in-flight answers so completed-message controls stay stable."""
    pending_records = [
        (metadata, snapshot)
        for metadata, snapshot in active_job_records(workspace_id=workspace_id)
        if metadata.get("kind") == "qa"
        and metadata.get("video_id") == video_id
        and snapshot.status in RUNNING_JOB_STATUSES
    ]
    for metadata, snapshot in pending_records:
        with st.chat_message("user"):
            st.markdown(str(metadata.get("question", "")))
        with st.chat_message("assistant"):
            st.markdown("**模型正在处理这个问题……**")
            st.progress(snapshot.progress, text=snapshot.message)
            st.caption(f"⏱️ {job_timing_text(snapshot)}")
            st.caption("这里显示检索、分段和生成等可观察步骤，不展示模型内部隐性思维链。")


def video_qa_panel(transcript: Transcript, settings: LLMSettings) -> None:
    """Keep completed controls stable and refresh only in-flight messages."""
    st.markdown(
        """
        <style>
        div[class*="st-key-edit_qa_"] {
            width: fit-content;
            margin-left: auto;
            margin-top: -.3rem;
        }
        div[class*="st-key-edit_qa_"] button {
            width: 2rem !important;
            min-width: 2rem !important;
            height: 2rem !important;
            min-height: 2rem !important;
            padding: 0 !important;
            border-radius: .45rem !important;
            font-size: .9rem !important;
            border-color: transparent !important;
            background: transparent !important;
        }
        div[class*="st-key-edit_qa_"] button p {
            display: none !important;
        }
        div[class*="st-key-edit_qa_"] button:hover {
            background: rgba(128, 128, 128, .12) !important;
        }
        div[class*="st-key-qa_edit_shell_"] [data-testid="stForm"] {
            border: 0 !important;
            padding: 0 !important;
        }
        div[class*="st-key-qa_edit_shell_"] button {
            width: 2rem !important;
            min-width: 2rem !important;
            height: 2rem !important;
            min-height: 2rem !important;
            padding: 0 !important;
            border-radius: 999px !important;
        }
        div[class*="st-key-qa_edit_shell_"] button p {
            font-size: 0 !important;
        }
        div[class*="st-key-previous_qa_branch_"] button,
        div[class*="st-key-next_qa_branch_"] button {
            min-height: 1.8rem !important;
            height: 1.8rem !important;
            padding: 0 .45rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    video_id = transcript.video_id
    pending_records = [
        (metadata, snapshot)
        for metadata, snapshot in active_job_records(
            workspace_id=str(st.session_state.workspace_id)
        )
        if metadata.get("kind") == "qa"
        and metadata.get("video_id") == video_id
        and snapshot.status in RUNNING_JOB_STATUSES
    ]
    branches, active_branch_id = normalize_chat_branches(
        video_id, st.session_state.chat_history
    )
    active_branch_index = next(
        (
            index
            for index, branch in enumerate(branches)
            if str(branch.get("id")) == active_branch_id
        ),
        len(branches) - 1,
    )
    active_branch = branches[active_branch_index]
    fork_index = active_branch.get("fork_index")
    editing_index = st.session_state.qa_editing.get(video_id)
    queued_edit = st.session_state.get("pending_qa_edit")
    if isinstance(queued_edit, dict) and queued_edit.get("video_id") == video_id:
        st.session_state.pending_qa_edit = None
        queued_index = int(queued_edit.get("index", -1))
        queued_branch_id = str(queued_edit.get("branch_id", ""))
        edited_question = str(queued_edit.get("question", "")).strip()
        if queued_branch_id != active_branch_id or not (
            0 <= queued_index < len(st.session_state.chat_history)
        ):
            errors = dict(st.session_state.qa_edit_errors)
            errors[video_id] = "当前对话版本已经变化，请重新点击铅笔修改。"
            st.session_state.qa_edit_errors = errors
        else:
            issue = validate_llm_settings(settings)
            if issue:
                open_settings_with_error(issue)
                st.session_state.ai_error = {"scope": "qa", "message": issue}
                st.rerun()
            prefix = list(st.session_state.chat_history[:queued_index])
            st.session_state.video_chat_branches[video_id] = branches
            st.session_state.video_active_chat_branches[video_id] = active_branch_id
            st.session_state.video_chats[video_id] = prefix
            st.session_state.chat_history = prefix
            cancel_question_edit(video_id)
            schedule_ai_job(
                "qa",
                transcript,
                settings,
                question=edited_question,
                history_override=prefix,
                edit_index=queued_index,
                new_branch_id=f"branch-{uuid4().hex}",
            )

    for index, item in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            if editing_index == index:
                with st.container(
                    key=f"qa_edit_shell_{video_id}_{active_branch_id}_{index}"
                ):
                    edit_widget_key = (
                        f"qa_edit_text_{video_id}_{active_branch_id}_{index}"
                    )
                    with st.form(
                        f"edit_qa_form_{video_id}_{active_branch_id}_{index}"
                    ):
                        edited_question = st.text_area(
                            "修改问题",
                            value=str(item["question"]),
                            key=edit_widget_key,
                            label_visibility="collapsed",
                        )
                        button_columns = st.columns([8, 1.2, 1.2])
                        cancel_edit = button_columns[1].form_submit_button(
                            "取消修改",
                            icon=":material/close:",
                            help="取消修改并恢复原问题",
                            type="tertiary",
                            on_click=cancel_question_edit,
                            args=(video_id,),
                        )
                        button_columns[2].form_submit_button(
                            "发送修改",
                            icon=":material/arrow_upward:",
                            help="发送修改后的问题",
                            type="primary",
                            on_click=queue_question_edit,
                            args=(
                                video_id,
                                index,
                                active_branch_id,
                                edit_widget_key,
                            ),
                        )
                edit_error = st.session_state.qa_edit_errors.get(video_id)
                if edit_error:
                    st.warning(edit_error)
            else:
                st.markdown(item["question"])
                action_columns = st.columns([0.8, 1, 0.8, 8])
                if len(branches) > 1 and index == fork_index:
                    if action_columns[0].button(
                        "‹",
                        key=f"previous_qa_branch_{video_id}_{active_branch_id}",
                        help="查看上一版本",
                        disabled=active_branch_index == 0 or bool(pending_records),
                    ):
                        select_chat_branch(
                            video_id, str(branches[active_branch_index - 1]["id"])
                        )
                        st.rerun()
                    action_columns[1].caption(
                        f"{active_branch_index + 1} / {len(branches)}"
                    )
                    if action_columns[2].button(
                        "›",
                        key=f"next_qa_branch_{video_id}_{active_branch_id}",
                        help="查看下一版本",
                        disabled=(
                            active_branch_index >= len(branches) - 1
                            or bool(pending_records)
                        ),
                    ):
                        select_chat_branch(
                            video_id, str(branches[active_branch_index + 1]["id"])
                        )
                        st.rerun()
                action_columns[3].button(
                    "修改",
                    key=f"edit_qa_{video_id}_{active_branch_id}_{index}",
                    icon=":material/edit:",
                    help="修改问题",
                    disabled=bool(pending_records),
                    on_click=begin_question_edit,
                    args=(video_id, index),
                )
        with st.chat_message("assistant"):
            st.markdown(item["answer"])

    pending_video_qa_messages(video_id, str(st.session_state.workspace_id))

    question = st.chat_input(
        "询问这个视频，例如：作者的核心观点是什么？",
        key=f"qa_input_{video_id}",
        disabled=interaction_busy(),
    )
    if question:
        schedule_ai_job("qa", transcript, settings, question=question)
    show_ai_error("qa")


def show_transcript(transcript: Transcript, settings: LLMSettings) -> None:
    if st.session_state.current_video_id != transcript.video_id:
        activate_transcript(transcript)
    duration = transcript.segments[-1].end if transcript.segments else 0
    col1, col2 = st.columns(2)
    col1.metric("字幕段数", len(transcript.segments))
    col2.metric("时长", format_timestamp(duration))

    visual_notice = st.session_state.visual_accuracy_notices.get(transcript.video_id)
    if not isinstance(visual_notice, dict) and "画面分析" in transcript.source:
        if "Ollama" in transcript.source:
            message = (
                "这份内容包含本地模型生成的画面识别结果。本地视觉模型的准确率可能受"
                "模型规模、量化方式和本机性能影响，请谨慎核对人物、画面文字、数字和关键情节。"
            )
            visual_notice = {"message": message, "local": True}
        else:
            message = (
                f"这份内容包含 API 模型生成的画面识别结果（{transcript.source}）。"
                "准确率取决于当时所用模型的视觉能力，请结合原视频核对关键内容。"
            )
            visual_notice = {"message": message, "local": False}
    if isinstance(visual_notice, dict):
        if visual_notice.get("local"):
            st.warning(str(visual_notice.get("message", "")))
        else:
            st.info(str(visual_notice.get("message", "")))

    with st.expander("查看完整字幕"):
        text = transcript_as_text(transcript.segments)
        st.text_area("字幕", text, height=360, label_visibility="collapsed")
        st.download_button(
            "下载 TXT",
            data=text.encode("utf-8"),
            file_name=f"{transcript.video_id}-transcript.txt",
            mime="text/plain",
        )

    active_view = st.radio(
        "阅读功能",
        ["内容总结", "视频问答"],
        horizontal=True,
        label_visibility="collapsed",
        key="active_view",
        disabled=interaction_busy(),
    )
    if active_view == "内容总结":
        st.caption("默认生成覆盖完整时间线、论证过程、例子和关键细节的详细视频笔记。")
        if st.button(
            "生成详细视频笔记",
            type="primary",
            use_container_width=True,
            disabled=interaction_busy(),
        ):
            schedule_ai_job("summary", transcript, settings)
        show_ai_error("summary")
        if st.session_state.summary:
            st.markdown(st.session_state.summary)

    else:
        video_qa_panel(transcript, settings)


initialize_state()
restore_background_job_state()
settings = st.session_state.llm_settings
reader_settings = st.session_state.reader_settings

with st.sidebar:
    st.markdown("### 📺 B站 AI 阅读器")
    st.button(
        "＋ 新建对话",
        type="primary",
        use_container_width=True,
        key="new_conversation",
        disabled=interaction_busy(),
        on_click=start_new_conversation,
        help="清空当前工作区以读取新链接；不会删除历史、字幕、音频或已下载视频。",
    )
    history_sidebar(st.container(), disabled=interaction_busy())
    background_ai_panel()

title_col, menu_col = st.columns([0.86, 0.14], vertical_alignment="center")
with title_col:
    st.title("📺 B站视频 AI 阅读器")
with menu_col:
    open_settings_clicked = st.button(
        "⚙ 设置",
        key="top_right_menu",
        help=f"设置 · 当前模型：{settings.provider} / {settings.model or '未选择'}",
        use_container_width=True,
    )
if open_settings_clicked:
    prepare_settings_widgets()
    st.session_state.settings_dialog_open = False
    settings_dialog()
elif st.session_state.settings_dialog_open:
    st.session_state.settings_dialog_open = False
    settings_dialog()

if st.session_state.restore_dialog_open:
    restore_history_dialog()

st.caption("读取顺序：已有字幕 → 音频 Whisper 转写 → 无可用文字时按时间抽取画面分析。")
if st.session_state.flash_message:
    st.toast(st.session_state.flash_message)
    st.session_state.flash_message = ""

current_conversation_progress_panel()

url = st.text_input(
    "B站视频链接或 BV 号",
    placeholder="https://www.bilibili.com/video/BV...",
    key="video_url",
    disabled=interaction_busy(),
)
if st.button(
    "读取视频信息",
    type="primary",
    disabled=interaction_busy() or not url.strip(),
):
    schedule_video_inspection(url)

parts = st.session_state.parts
if parts:
    part_index = st.selectbox(
        "选择分P",
        range(len(parts)),
        format_func=lambda index: f"P{parts[index]['index']} · {parts[index]['title']}",
        disabled=interaction_busy(),
    )
    part = parts[part_index]
    if part.get("thumbnail"):
        st.image(part["thumbnail"], width=320)
    st.subheader(part["title"])
    st.caption(f"视频 ID：{part['id']} · 时长：{format_timestamp(part['duration'])}")

    cached = load_transcript(part["id"])
    if cached and (
        st.session_state.transcript is None
        or st.session_state.transcript.video_id != part["id"]
    ):
        activate_transcript(cached)
        st.caption(f"已载入本地缓存：{cached.source}")

    tracks = part.get("tracks") or []
    has_audio = part.get("has_audio")
    audio_failures = st.session_state.audio_failures
    audio_failure = str(audio_failures.get(part["id"], ""))
    route_text = [f"检测到 {len(tracks)} 条字幕" if tracks else "没有检测到字幕"]
    if has_audio is True:
        route_text.append("检测到音轨")
    elif has_audio is False:
        route_text.append("没有检测到音轨")
    else:
        route_text.append("音轨状态将在下载时确认")
    st.caption(" · ".join(route_text) + "。一键读取会自动选择可用路线。")
    read_button_label = "🔄 重新读取视频内容" if cached else "✨ 一键智能读取"
    if st.button(
        read_button_label,
        type="primary",
        use_container_width=True,
        disabled=interaction_busy(),
        key=f"smart_read_{part['id']}",
        help=(
            "重新检测字幕和音轨，按当前设置重新生成内容并覆盖旧缓存。"
            if cached
            else "依次尝试现有字幕、音频转写；两者都不可用时才使用支持图片输入的模型分析画面。"
        ),
    ):
        if not tracks and has_audio is False:
            issue = validate_llm_settings(settings)
            support, support_message = vision_support_status(settings)
            if issue:
                open_settings_with_error(issue)
                st.rerun()
            if support == "unsupported":
                open_settings_with_error(support_message)
                st.rerun()
        schedule_media_job("smart", part, track=tracks[0] if tracks else None)

    show_advanced_reading = st.toggle(
        "显示高级读取选项",
        value=False,
        key=f"show_advanced_reading_{part['id']}",
        disabled=interaction_busy(),
        help="需要手动指定字幕、强制重新转写或单独使用画面分析时再打开。",
    )

    if tracks and show_advanced_reading:
        selected_track_index = st.selectbox(
            "可用字幕",
            range(len(tracks)),
            format_func=lambda index: (
                f"{tracks[index]['name']} · {tracks[index]['category']} · {tracks[index]['ext']}"
            ),
            disabled=interaction_busy(),
        )
        if st.button(
            "提取所选字幕",
            use_container_width=True,
            disabled=interaction_busy(),
            key=f"extract_subtitle_{part['id']}",
        ):
            schedule_media_job("subtitle", part, track=tracks[selected_track_index])
    elif not tracks:
        if has_audio is False:
            st.warning("没有检测到字幕，也没有检测到音轨；程序已跳过 Whisper，下一步可以读取视频画面。")
        else:
            st.warning("没有检测到可用字幕。下一步可以使用视频音轨进行 Whisper 转写。")

    if has_audio is not False and show_advanced_reading:
        with st.container(border=True, key=f"audio_fallback_{part['id']}"):
            st.markdown("#### 🎙️ 使用音频转成文字")
            st.caption(
                "仅在没有字幕或字幕质量不理想时使用。音频保存在本机 `data` 目录；"
                "Whisper 模型首次使用时会自动下载。"
            )
            if st.button(
                "下载音频并开始转写",
                use_container_width=True,
                disabled=interaction_busy(),
                key=f"transcribe_audio_{part['id']}",
            ):
                schedule_media_job("audio", part)

    visual_reason = ""
    if has_audio is False:
        visual_reason = "视频元数据表明这个分P没有音轨。"
    elif audio_failure:
        visual_reason = f"音频路线不可用：{audio_failure}"

    if visual_reason and show_advanced_reading:
        with st.container(border=True, key=f"visual_fallback_{part['id']}"):
            st.markdown("#### 👁️ 改用视频画面分析")
            st.caption(visual_reason)
            planned_frames, frame_interval = frame_sampling_plan(float(part.get("duration") or 0))
            st.write(
                f"程序将下载最高约 720p 的视频，预计抽取约 **{planned_frames} 张**画面，"
                f"平均约 **{frame_interval:.0f} 秒一张**，再识别场景、动作、图表和画面文字。"
            )
            support, support_message = vision_support_status(settings)
            if support == "supported":
                st.success(f"视觉能力检查：{support_message}")
            elif support == "unsupported":
                st.error(f"视觉能力检查：{support_message}")
            else:
                st.warning(f"视觉能力检查：{support_message}")
            accuracy_notice, local_visual_model = visual_accuracy_notice(settings)
            if local_visual_model:
                st.warning(accuracy_notice)
            else:
                st.info(accuracy_notice)
            st.caption(
                "自适应抽帧比原来的固定 24 帧更密集，但仍可能漏掉一闪而过的内容，也无法还原声音。"
                "云端模型会接收截图并可能产生图片输入费用；最长视频最多抽取 180 帧。"
            )
            if support == "unsupported" and st.button(
                "打开设置选择视觉模型",
                use_container_width=True,
                key=f"open_vision_settings_{part['id']}",
                disabled=interaction_busy(),
            ):
                open_settings_with_error("当前模型不支持图片输入，请选择明确支持视觉/图片的模型。")
                st.rerun()
            if st.button(
                "下载视频并读取画面",
                use_container_width=True,
                disabled=interaction_busy() or support == "unsupported",
                key=f"analyze_frames_{part['id']}",
            ):
                issue = validate_llm_settings(settings)
                if issue:
                    open_settings_with_error(issue)
                    st.rerun()
                schedule_media_job("visual", part)

    transcript = st.session_state.transcript
    if transcript and transcript.video_id == part["id"]:
        st.divider()
        show_transcript(transcript, settings)
elif st.session_state.transcript:
    transcript = st.session_state.transcript
    st.subheader(transcript.title)
    st.caption(f"视频 ID：{transcript.video_id} · 从本地内容历史打开")
    if st.button(
        "🔄 重新读取视频内容",
        type="primary",
        use_container_width=True,
        key=f"reread_history_{transcript.video_id}",
        help="重新检测该视频的字幕和音轨，并按当前设置重新生成内容、覆盖旧缓存。",
    ):
        duration = transcript.segments[-1].end if transcript.segments else 0
        schedule_media_job(
            "reread",
            {
                "id": transcript.video_id,
                "index": 1,
                "title": transcript.title,
                "duration": duration,
                "url": transcript.video_id,
                "thumbnail": "",
                "tracks": [],
                "has_audio": None,
            },
        )
    st.divider()
    show_transcript(transcript, settings)
