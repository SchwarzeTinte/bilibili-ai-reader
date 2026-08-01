from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

import requests

from .models import Segment, Transcript
from .text import format_timestamp, transcript_as_text


@dataclass(slots=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""


def _generate(
    settings: LLMSettings,
    system: str,
    prompt: str,
    max_tokens: int = 1_200,
) -> str:
    if settings.provider == "Gemini":
        if not settings.api_key:
            raise ValueError("请填写 Gemini API Key。")
        from google import genai

        client = genai.Client(api_key=settings.api_key)
        response = client.models.generate_content(
            model=settings.model,
            contents=f"{system}\n\n{prompt}",
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini 没有返回文本内容。")
        return text

    if settings.provider == "Ollama":
        host = (settings.base_url or "http://localhost:11434/v1").rstrip("/")
        if host.endswith("/v1"):
            host = host[:-3]
        try:
            response = requests.post(
                f"{host}/api/chat",
                json={
                    "model": settings.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "think": False,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_ctx": 8_192,
                        "num_predict": max_tokens,
                    },
                },
                timeout=300,
            )
            if response.status_code == 404:
                raise ValueError(
                    f"Ollama 中没有模型 `{settings.model}`。请从左侧选择已经安装的本机模型。"
                )
            response.raise_for_status()
            content = str(response.json().get("message", {}).get("content", "")).strip()
        except requests.Timeout as exc:
            raise RuntimeError("Ollama 响应超时。请换用更小的模型，或稍后重试。") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"无法连接 Ollama：{exc}") from exc
        if not content:
            raise RuntimeError("Ollama 没有返回文本内容。")
        return content

    from openai import OpenAI

    if not settings.api_key:
        raise ValueError("请填写 API Key。")
    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url or None,
        timeout=180,
    )
    response = client.chat.completions.create(
        model=settings.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("模型没有返回文本内容。")
    return content


def _chunk_segments(segments: Iterable[Segment], max_chars: int = 12_000) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for segment in segments:
        line = f"[{format_timestamp(segment.start)}] {segment.text}"
        if current and current_size + len(line) + 1 > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_size = 0
        current.append(line)
        current_size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def summarize(
    transcript: Transcript,
    settings: LLMSettings,
    progress: Callable[[int, str], None] | None = None,
) -> str:
    chunk_size = 5_000 if settings.provider == "Ollama" else 12_000
    chunks = _chunk_segments(transcript.segments, max_chars=chunk_size)
    system = (
        "你是一名严谨的视频内容分析助手。只能依据给定字幕作答，不要虚构画面或事实。"
        "使用简体中文，并用 [MM:SS] 或 [HH:MM:SS] 标注关键结论的依据。"
    )
    if len(chunks) == 1:
        if progress:
            progress(10, "正在生成视频摘要……")
        result = _generate(
            settings,
            system,
            "请整理以下视频字幕，输出：\n"
            "1. 三至五句话的核心摘要\n2. 按时间排列的章节\n"
            "3. 关键观点或知识点\n4. 字幕中存在的不确定信息（如有）\n\n"
            f"视频标题：{transcript.title}\n\n字幕：\n{chunks[0]}",
            max_tokens=700,
        )
        if progress:
            progress(100, "摘要生成完成。")
        return result

    partials: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        if progress:
            percent = 5 + int(70 * (index - 1) / len(chunks))
            progress(percent, f"正在整理字幕第 {index}/{len(chunks)} 段……")
        partials.append(
            _generate(
                settings,
                system,
                f"这是视频字幕的第 {index}/{len(chunks)} 段。提炼本段要点，保留时间戳，"
                f"不要下超出本段的信息。\n\n{chunk}",
                max_tokens=220,
            )
        )
    combined = "\n\n".join(f"第 {i} 段摘要：\n{text}" for i, text in enumerate(partials, start=1))
    if progress:
        progress(80, "分段整理完成，正在合并最终摘要……")
    result = _generate(
        settings,
        system,
        "请合并下面的分段摘要，删除重复内容，输出完整的核心摘要、章节和关键观点。"
        "保留准确时间戳。\n\n" + combined,
        max_tokens=700,
    )
    if progress:
        progress(100, "摘要生成完成。")
    return result


def _question_terms(question: str) -> set[str]:
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", question.lower())
    terms = {normalized[index : index + 2] for index in range(max(0, len(normalized) - 1))}
    terms.update(token for token in re.findall(r"[a-z0-9_]{3,}", normalized))
    return terms


def _relevant_context(transcript: Transcript, question: str, max_chars: int = 48_000) -> str:
    full_text = transcript_as_text(transcript.segments)
    if len(full_text) <= max_chars:
        return full_text
    chunks = _chunk_segments(transcript.segments, max_chars=6_000)
    terms = _question_terms(question)

    def score(chunk: str) -> int:
        lowered = chunk.lower()
        return sum(lowered.count(term) for term in terms)

    ranked = sorted(enumerate(chunks), key=lambda pair: (score(pair[1]), -pair[0]), reverse=True)
    selected_indices = sorted(index for index, _ in ranked[:8])
    return "\n\n".join(chunks[index] for index in selected_indices)


def answer_question(
    transcript: Transcript,
    question: str,
    settings: LLMSettings,
    history: list[dict[str, str]] | None = None,
) -> str:
    context_limit = 5_000 if settings.provider == "Ollama" else 48_000
    context = _relevant_context(transcript, question, max_chars=context_limit)
    recent_history = (history or [])[-4:]
    history_text = "\n".join(
        f"用户：{item.get('question', '')}\n助手：{item.get('answer', '')}" for item in recent_history
    )
    system = (
        "你是一名严谨的视频问答助手。只依据提供的字幕回答。"
        "找不到答案时明确说字幕中没有足够信息，不要猜测。"
        "尽量在结论后引用字幕已有的 [MM:SS] 或 [HH:MM:SS] 时间戳。"
    )
    prompt = f"视频标题：{transcript.title}\n"
    if history_text:
        prompt += f"\n最近对话：\n{history_text}\n"
    prompt += f"\n问题：{question}\n\n相关字幕：\n{context}"
    return _generate(settings, system, prompt, max_tokens=700)
