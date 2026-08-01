from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bili_reader.extractor import (  # noqa: E402
    BilibiliReaderError,
    download_audio,
    download_subtitle,
    inspect_video,
)
from bili_reader.llm import LLMSettings, answer_question, summarize  # noqa: E402
from bili_reader.models import Transcript  # noqa: E402
from bili_reader.storage import load_transcript, save_transcript  # noqa: E402
from bili_reader.text import format_timestamp, transcript_as_text  # noqa: E402
from bili_reader.transcriber import transcribe_audio  # noqa: E402


st.set_page_config(page_title="B站视频 AI 阅读器", page_icon="📺", layout="wide")


def initialize_state() -> None:
    defaults = {
        "parts": [],
        "transcript": None,
        "summary": "",
        "chat_history": [],
        "inspected_url": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def llm_settings_panel() -> LLMSettings:
    st.sidebar.subheader("AI 设置")
    provider = st.sidebar.selectbox("模型服务", ["Gemini", "DeepSeek", "OpenAI", "Ollama"])
    defaults = {
        "Gemini": ("gemini-2.5-flash", ""),
        "DeepSeek": ("deepseek-chat", "https://api.deepseek.com"),
        "OpenAI": ("gpt-4.1-mini", "https://api.openai.com/v1"),
        "Ollama": ("qwen3:8b", "http://localhost:11434/v1"),
    }
    default_model, default_url = defaults[provider]
    model = st.sidebar.text_input("模型名称", value=default_model, key=f"model_{provider}")
    if provider == "Ollama":
        api_key = ""
        base_url = st.sidebar.text_input("接口地址", value=default_url)
        st.sidebar.caption("Ollama 完全在本机运行，无需 API Key。")
    else:
        api_key = st.sidebar.text_input("API Key（仅保存在当前会话）", type="password")
        base_url = "" if provider == "Gemini" else st.sidebar.text_input("接口地址", value=default_url)
    return LLMSettings(provider=provider, model=model.strip(), api_key=api_key.strip(), base_url=base_url.strip())


def show_transcript(transcript: Transcript, settings: LLMSettings) -> None:
    duration = transcript.segments[-1].end if transcript.segments else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("字幕段数", len(transcript.segments))
    col2.metric("时长", format_timestamp(duration))
    col3.metric("来源", transcript.source)

    with st.expander("查看完整字幕"):
        text = transcript_as_text(transcript.segments)
        st.text_area("字幕", text, height=360, label_visibility="collapsed")
        st.download_button(
            "下载 TXT",
            data=text.encode("utf-8"),
            file_name=f"{transcript.video_id}-transcript.txt",
            mime="text/plain",
        )

    summary_tab, chat_tab = st.tabs(["内容总结", "视频问答"])
    with summary_tab:
        if st.button("生成总结", type="primary", use_container_width=True):
            try:
                with st.spinner("AI 正在整理字幕……"):
                    st.session_state.summary = summarize(transcript, settings)
            except Exception as exc:
                st.error(f"总结失败：{exc}")
        if st.session_state.summary:
            st.markdown(st.session_state.summary)

    with chat_tab:
        for item in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(item["question"])
            with st.chat_message("assistant"):
                st.markdown(item["answer"])
        question = st.chat_input("询问这个视频，例如：作者的核心观点是什么？")
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            try:
                with st.chat_message("assistant"):
                    with st.spinner("正在查找字幕并回答……"):
                        answer = answer_question(
                            transcript,
                            question,
                            settings,
                            st.session_state.chat_history,
                        )
                    st.markdown(answer)
                st.session_state.chat_history.append({"question": question, "answer": answer})
            except Exception as exc:
                st.error(f"问答失败：{exc}")


initialize_state()
settings = llm_settings_panel()

st.title("📺 B站视频 AI 阅读器")
st.caption("优先读取已有字幕；没有字幕时，在本机下载音频并用 Whisper 转写。")

with st.sidebar:
    st.subheader("本地读取设置")
    auth_mode = st.selectbox("B站登录状态（按需）", ["不使用", "读取浏览器", "cookies.txt 文件"])
    auth_source = None
    if auth_mode == "读取浏览器":
        browser_label = st.selectbox("浏览器", ["Edge", "Chrome", "Firefox"])
        auth_source = f"browser:{browser_label}"
        st.caption("Windows 可能锁定浏览器 Cookie 数据库；读取时请完全退出所选浏览器。")
    elif auth_mode == "cookies.txt 文件":
        cookie_path = st.text_input("cookies.txt 的本机绝对路径")
        if cookie_path.strip():
            auth_source = f"file:{cookie_path.strip().strip(chr(34))}"
    whisper_model = st.selectbox(
        "Whisper 模型",
        ["tiny", "base", "small", "medium", "large-v3"],
        index=2,
        help="模型越大越准确，也越慢。普通电脑建议 small。",
    )
    whisper_device = st.selectbox(
        "Whisper 运行设备",
        ["CPU（推荐）", "自动检测 GPU"],
        help="CPU 兼容性最好；只有完整安装 NVIDIA CUDA 运行库时才建议自动检测 GPU。",
    )
    whisper_language = st.selectbox("语音语言", ["中文", "自动检测", "英文"])
    st.info(
        "Cookie 只由 yt-dlp 用于向 B站发起请求，不会发送给 AI 服务，也不会由本程序保存。"
        "选择云端 AI 时，字幕文本会发送给相应服务；需要完全本地处理请选择 Ollama。"
    )

url = st.text_input("B站视频链接或 BV 号", placeholder="https://www.bilibili.com/video/BV...")
if st.button("读取视频信息", type="primary", disabled=not url.strip()):
    try:
        with st.spinner("正在读取视频信息和字幕列表……"):
            parts = inspect_video(url.strip(), auth_source)
        st.session_state.parts = parts
        st.session_state.inspected_url = url.strip()
        st.session_state.transcript = None
        st.session_state.summary = ""
        st.session_state.chat_history = []
        st.success(f"读取成功，共发现 {len(parts)} 个分P。")
    except Exception as exc:
        st.error(str(exc))

parts = st.session_state.parts
if parts:
    part_index = st.selectbox(
        "选择分P",
        range(len(parts)),
        format_func=lambda index: f"P{parts[index]['index']} · {parts[index]['title']}",
    )
    part = parts[part_index]
    if part.get("thumbnail"):
        st.image(part["thumbnail"], width=320)
    st.subheader(part["title"])
    st.caption(f"视频 ID：{part['id']} · 时长：{format_timestamp(part['duration'])}")

    cached = load_transcript(part["id"])
    if cached and st.session_state.transcript is None:
        st.session_state.transcript = cached
        st.info(f"已载入本地缓存：{cached.source}")

    tracks = part.get("tracks") or []
    if tracks:
        selected_track_index = st.selectbox(
            "可用字幕",
            range(len(tracks)),
            format_func=lambda index: (
                f"{tracks[index]['name']} · {tracks[index]['category']} · {tracks[index]['ext']}"
            ),
        )
        if st.button("提取所选字幕", use_container_width=True):
            try:
                with st.spinner("正在下载并解析字幕……"):
                    transcript = download_subtitle(part, tracks[selected_track_index])
                    save_transcript(transcript)
                st.session_state.transcript = transcript
                st.session_state.summary = ""
                st.session_state.chat_history = []
                st.success("字幕提取完成。")
            except BilibiliReaderError as exc:
                st.error(str(exc))
    else:
        st.warning("这个分P没有检测到可用字幕，可以在本机下载音频并转写。")

    with st.expander("没有字幕或字幕质量差？使用 Whisper 转写"):
        st.write("音频保存在项目的 `data` 目录；Whisper 模型首次使用时会下载到本机缓存。")
        if st.button("下载音频并开始转写", use_container_width=True):
            progress_bar = st.progress(0, text="准备下载音频……")
            try:
                audio_path = download_audio(part, auth_source)
                progress_bar.progress(2, text="音频下载完成，正在准备 Whisper……")
                language_map = {"中文": "zh", "自动检测": None, "英文": "en"}

                def update_progress(percent: int, message: str) -> None:
                    progress_bar.progress(percent, text=message)

                transcript = transcribe_audio(
                    audio_path,
                    video_id=part["id"],
                    title=part["title"],
                    model_size=whisper_model,
                    language=language_map[whisper_language],
                    device_mode="cpu" if whisper_device == "CPU（推荐）" else "auto",
                    progress=update_progress,
                )
                save_transcript(transcript)
                st.session_state.transcript = transcript
                st.session_state.summary = ""
                st.session_state.chat_history = []
                st.success("转写完成并已缓存到本机。")
            except Exception as exc:
                st.error(f"处理失败：{exc}")

    transcript = st.session_state.transcript
    if transcript and transcript.video_id == part["id"]:
        st.divider()
        show_transcript(transcript, settings)
