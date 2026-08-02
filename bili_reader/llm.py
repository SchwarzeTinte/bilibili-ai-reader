from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import requests

from .jobs import JobCancelledError, check_current_job_cancelled
from .models import Segment, Transcript
from .text import format_timestamp, transcript_as_text


@dataclass(slots=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""
    context_window: int = 8_192
    max_context_window: int | None = None


@dataclass(slots=True)
class _GenerationResult:
    text: str
    truncated: bool = False


def _stopped_for_length(reason: object) -> bool:
    if reason is None:
        return False
    normalized = " ".join(
        str(value)
        for value in (
            getattr(reason, "name", ""),
            getattr(reason, "value", ""),
            reason,
        )
        if value
    ).lower()
    return any(marker in normalized for marker in ("max_token", "max token", "length"))


def _output_budget(settings: LLMSettings, desired_tokens: int) -> int:
    """Keep enough room for the input while allowing useful answers on larger models."""
    return max(256, min(desired_tokens, max(256, settings.context_window // 4)))


def _estimate_text_tokens(text: str) -> int:
    """Conservatively estimate mixed Chinese/Latin text without a model tokenizer."""
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", text))
    other_count = max(0, len(text) - cjk_count)
    return cjk_count + (other_count + 3) // 4


def _ollama_runtime_context(
    settings: LLMSettings,
    system: str,
    prompt: str,
    max_tokens: int,
) -> int:
    """Allocate only the context this request needs instead of Ollama's theoretical max."""
    selected_limit = max(2_048, settings.context_window)
    required = _estimate_text_tokens(f"{system}\n\n{prompt}") + max_tokens + 512
    rounded = max(2_048, ((required + 2_047) // 2_048) * 2_048)
    return min(selected_limit, rounded)


def _ollama_error_detail(response: requests.Response) -> str:
    try:
        detail = response.json().get("error", "")
    except (requests.JSONDecodeError, ValueError, AttributeError):
        detail = response.text
    if isinstance(detail, str) and detail.startswith("{"):
        try:
            nested = json.loads(detail)
            detail = nested.get("error", {}).get("message", detail)
        except (json.JSONDecodeError, AttributeError):
            pass
    return str(detail).strip() or response.reason or "未知错误"


def _openai_chat_with_fallback(client, options: dict[str, object]):
    request_options = dict(options)
    for _ in range(5):
        try:
            return client.chat.completions.create(**request_options)
        except Exception as exc:
            message = str(exc).lower()
            if "temperature" in message and "temperature" in request_options:
                request_options.pop("temperature")
                continue
            if "max_tokens" in message and "max_tokens" in request_options:
                request_options["max_completion_tokens"] = request_options.pop("max_tokens")
                continue
            if (
                ("thinking" in message or "extra_body" in message)
                and "extra_body" in request_options
            ):
                request_options.pop("extra_body")
                continue
            if "system" in message and len(request_options.get("messages", [])) == 2:
                messages = request_options["messages"]
                request_options["messages"] = [
                    {
                        "role": "user",
                        "content": f"{messages[0]['content']}\n\n{messages[1]['content']}",
                    }
                ]
                continue
            raise
    return client.chat.completions.create(**request_options)


def _generate_once(
    settings: LLMSettings,
    system: str,
    prompt: str,
    max_tokens: int = 1_200,
) -> _GenerationResult:
    if not settings.model.strip():
        raise ValueError("请先选择或填写模型名称。")

    if settings.provider == "Gemini":
        if not settings.api_key:
            raise ValueError("请填写 Gemini API Key。")
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.api_key)
        response = client.models.generate_content(
            model=settings.model,
            contents=f"{system}\n\n{prompt}",
            config=types.GenerateContentConfig(max_output_tokens=max_tokens),
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini 没有返回文本内容。")
        candidates = getattr(response, "candidates", None) or []
        finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        return _GenerationResult(str(text).strip(), _stopped_for_length(finish_reason))

    if settings.provider == "Anthropic":
        if not settings.api_key:
            raise ValueError("请填写 Anthropic API Key。")
        from anthropic import Anthropic

        client = Anthropic(
            api_key=settings.api_key,
            base_url=settings.base_url or None,
            timeout=180,
        )
        response = client.messages.create(
            model=settings.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        content = "".join(
            str(getattr(block, "text", ""))
            for block in response.content
            if getattr(block, "type", "") == "text"
        ).strip()
        if not content:
            raise RuntimeError("Anthropic 没有返回文本内容。")
        return _GenerationResult(
            content,
            _stopped_for_length(getattr(response, "stop_reason", None)),
        )

    if settings.provider == "Ollama":
        host = (settings.base_url or "http://localhost:11434/v1").rstrip("/")
        if host.endswith("/v1"):
            host = host[:-3]
        runtime_context = _ollama_runtime_context(settings, system, prompt, max_tokens)
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
                    "stream": True,
                    "options": {
                        "temperature": 0.2,
                        "num_ctx": runtime_context,
                        "num_predict": max_tokens,
                    },
                },
                # Limit only the connection phase. Streaming produces data throughout a
                # long generation, so a fixed read timeout would incorrectly kill slow
                # CPU inference or a large-context prompt before it finishes.
                timeout=(10, None),
                stream=True,
            )
            if response.status_code == 404:
                raise ValueError(
                    f"Ollama 中没有模型 `{settings.model}`。请从左侧选择已经安装的本机模型。"
                )
            if not response.ok:
                detail = _ollama_error_detail(response)
                raise RuntimeError(f"Ollama 请求失败（HTTP {response.status_code}）：{detail}")
            content_parts: list[str] = []
            done_reason: object = None
            response_lines = (
                response.iter_lines()
                if getattr(response, "raw", None) is not None
                else response.content.splitlines()
            )
            for raw_line in response_lines:
                check_current_job_cancelled()
                if not raw_line:
                    continue
                try:
                    payload = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
                    raise RuntimeError("Ollama 返回了无法解析的流式响应。") from exc
                if payload.get("error"):
                    raise RuntimeError(f"Ollama 请求失败：{payload['error']}")
                content_parts.append(str(payload.get("message", {}).get("content", "")))
                if payload.get("done") or payload.get("done_reason"):
                    done_reason = payload.get("done_reason")
            content = "".join(content_parts).strip()
            truncated = _stopped_for_length(done_reason)
        except JobCancelledError:
            if "response" in locals():
                response.close()
            raise
        except requests.ConnectTimeout as exc:
            raise RuntimeError("连接 Ollama 超时。请确认服务已经启动，且接口地址可以访问。") from exc
        except requests.Timeout as exc:
            raise RuntimeError("Ollama 长时间没有返回数据。请减小上下文预算或换用更小的模型。") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"无法连接 Ollama：{exc}") from exc
        if not content:
            raise RuntimeError("Ollama 没有返回文本内容。")
        return _GenerationResult(content, truncated)

    from openai import OpenAI

    if settings.provider != "OpenAI 兼容（自定义）" and not settings.api_key:
        raise ValueError("请填写 API Key。")
    client = OpenAI(
        api_key=settings.api_key or "local-no-key-required",
        base_url=settings.base_url or None,
        timeout=180,
    )
    completion_options = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    if settings.provider == "DeepSeek":
        completion_options["extra_body"] = {"thinking": {"type": "disabled"}}
    response = _openai_chat_with_fallback(client, completion_options)
    choice = response.choices[0]
    content = choice.message.content
    if isinstance(content, list):
        content = "".join(
            str(block.get("text", "") if isinstance(block, dict) else getattr(block, "text", ""))
            for block in content
        )
    if not content:
        raise RuntimeError("模型没有返回文本内容。")
    return _GenerationResult(
        str(content).strip(),
        _stopped_for_length(getattr(choice, "finish_reason", None)),
    )


def _merge_continuation(existing: str, continuation: str) -> str:
    continuation = continuation.strip()
    if not continuation:
        return existing
    max_overlap = min(500, len(existing), len(continuation))
    for size in range(max_overlap, 19, -1):
        if existing[-size:] == continuation[:size]:
            continuation = continuation[size:].lstrip()
            break
    return f"{existing.rstrip()}\n{continuation}" if continuation else existing


def _generate(
    settings: LLMSettings,
    system: str,
    prompt: str,
    max_tokens: int = 1_200,
    continuation_progress: Callable[[int], None] | None = None,
) -> str:
    result = _generate_once(settings, system, prompt, max_tokens=max_tokens)
    text = result.text
    continuation_count = 0
    seen_continuations: set[str] = set()
    while result.truncated:
        continuation_count += 1
        if continuation_progress:
            continuation_progress(continuation_count)
        tail_limit = min(4_000, max(600, settings.context_window // 4))
        previous_tail = text[-tail_limit:]
        source_limit = max(
            200,
            settings.context_window - max_tokens - len(previous_tail) - 600,
        )
        source_excerpt = prompt if len(prompt) <= source_limit else prompt[-source_limit:]
        continuation_prompt = (
            "上一次回答因为达到输出长度上限而中断。请从中断处继续，只输出尚未完成的内容，"
            "不要重新写标题、不要重复已有段落，并把未完成的句子和列表写完整。\n\n"
            f"原始任务和材料（必要时仅保留后半部分）：\n{source_excerpt}\n\n"
            f"已经生成的末尾：\n{previous_tail}"
        )
        result = _generate_once(
            settings,
            system,
            continuation_prompt,
            max_tokens=max_tokens,
        )
        continuation = result.text.strip()
        signature = re.sub(r"\s+", "", continuation)[-1_000:]
        recent_text = re.sub(r"\s+", "", text[-max(8_000, len(continuation) + 500) :])
        if not signature or signature in seen_continuations or signature in recent_text:
            text += (
                "\n\n> 模型在自动续写时开始重复、没有产生新内容，程序已停止续写以避免无限循环。"
            )
            break
        seen_continuations.add(signature)
        merged = _merge_continuation(text, continuation)
        if len(merged) <= len(text):
            text += "\n\n> 模型在自动续写时没有产生新内容，程序已停止续写以避免无限循环。"
            break
        text = merged
    return text


def test_connection(settings: LLMSettings) -> str:
    """Send a minimal request to verify the selected provider and model."""
    return _generate(
        settings,
        "你是连接测试助手。",
        "只回复：连接成功",
        max_tokens=128,
    )


def vision_support_status(settings: LLMSettings) -> tuple[str, str]:
    """Return supported/unsupported/unknown with a user-facing explanation."""
    model = settings.model.strip().lower()
    if not model:
        return "unsupported", "尚未选择模型，无法进行图片分析。"
    vision_markers = ("vision", "vl", "llava", "minicpm-v", "qwen2.5-vl", "qwen3-vl")
    if settings.provider == "Ollama":
        host = (settings.base_url or "http://localhost:11434/v1").rstrip("/")
        if host.endswith("/v1"):
            host = host[:-3]
        try:
            response = requests.post(
                f"{host}/api/show",
                json={"model": settings.model},
                timeout=3,
            )
            response.raise_for_status()
            payload = response.json()
            capabilities = {
                str(item).lower() for item in payload.get("capabilities", [])
            }
            if "vision" in capabilities:
                return "supported", "Ollama 已确认该模型具有 vision（图片输入）能力。"
            if capabilities:
                return "unsupported", "Ollama 返回的模型能力中没有 vision；请更换视觉模型。"
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            pass
        if any(marker in model for marker in vision_markers):
            return "unknown", "模型名称像视觉模型，但 Ollama 未返回可确认的能力信息，将在运行时验证。"
        return "unknown", "Ollama 未能确认该模型是否支持图片；点击分析后会发送一批截图进行实际验证。"
    if settings.provider == "DeepSeek":
        if any(marker in model for marker in vision_markers):
            return "unknown", "模型名称像视觉模型，但仍需由当前 DeepSeek 接口实际确认图片输入能力。"
        return "unsupported", "当前 DeepSeek 文本模型配置未声明图片输入能力，请改用视觉模型或兼容接口。"
    if settings.provider == "Gemini":
        return "supported", "Gemini 内容生成模型通常支持图片输入，程序仍会在请求时进行最终验证。"
    if settings.provider == "Anthropic" and "claude" in model:
        return "supported", "当前 Claude 模型配置可尝试图片输入，程序仍会在请求时进行最终验证。"
    if settings.provider == "OpenAI" and any(
        marker in model for marker in ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4")
    ):
        return "supported", "当前 OpenAI 模型系列支持或通常支持图片输入，程序仍会在请求时最终验证。"
    if any(marker in model for marker in vision_markers):
        return "unknown", "模型名称像视觉模型，但自定义服务未公开统一能力字段，将在运行时验证。"
    return "unknown", "当前服务没有统一的视觉能力查询方式；图片分析请求会进行最终兼容性验证。"


def describe_images(
    settings: LLMSettings,
    images: list[tuple[str, Path]],
) -> str:
    """Describe timestamp-labelled frames using the selected multimodal model."""
    if not images:
        raise ValueError("没有可供分析的视频画面。")
    if not settings.model.strip():
        raise ValueError("请先选择支持图片输入的模型。")

    system = (
        "你是一名严谨的视频画面分析助手。只描述给定截图中能直接观察到的内容，不猜测截图之外的"
        "对白、声音、人物身份或事件因果。识别画面文字、图表、动作、场景变化和关键视觉信息。"
        "逐字列出画面中所有能够确认的文字；无法看清的文字必须写‘无法辨认’，不得跳过后自行补全。"
        "相似角色无法确认时只描述外观，禁止根据常识补全身份。"
    )
    prompt = (
        "这些图片按视频时间顺序排列，标签分别为："
        + "、".join(label for label, _ in images)
        + "。每个标签只对应紧随其后的那一张图片。请按时间标签逐项输出直接观察结果；"
        "不要把其他时间图片中的人物、文字或动作移到当前标签，也不要跳过任何标签。"
    )

    encoded: list[tuple[str, str, str]] = []
    for label, path in images:
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded.append((label, mime_type, base64.b64encode(path.read_bytes()).decode("ascii")))

    if settings.provider == "Gemini":
        if not settings.api_key:
            raise ValueError("请填写 Gemini API Key。")
        from google import genai
        from google.genai import types

        contents: list[object] = [f"{system}\n\n{prompt}"]
        for label, mime_type, data in encoded:
            contents.extend([label, types.Part.from_bytes(data=base64.b64decode(data), mime_type=mime_type)])
        response = genai.Client(api_key=settings.api_key).models.generate_content(
            model=settings.model,
            contents=contents,
            config=types.GenerateContentConfig(max_output_tokens=_output_budget(settings, 1_800)),
        )
        text = getattr(response, "text", None)
    elif settings.provider == "Anthropic":
        if not settings.api_key:
            raise ValueError("请填写 Anthropic API Key。")
        from anthropic import Anthropic

        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        for label, mime_type, data in encoded:
            content.extend(
                [
                    {"type": "text", "text": label},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime_type, "data": data},
                    },
                ]
            )
        response = Anthropic(
            api_key=settings.api_key,
            base_url=settings.base_url or None,
            timeout=180,
        ).messages.create(
            model=settings.model,
            max_tokens=_output_budget(settings, 1_800),
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(
            str(getattr(block, "text", ""))
            for block in response.content
            if getattr(block, "type", "") == "text"
        )
    elif settings.provider == "Ollama":
        host = (settings.base_url or "http://localhost:11434/v1").rstrip("/")
        if host.endswith("/v1"):
            host = host[:-3]
        messages: list[dict[str, object]] = [{"role": "system", "content": system}]
        for label, _, data in encoded:
            messages.append(
                {
                    "role": "user",
                    "content": f"只分析这一张图片，时间标签是 {label}。",
                    "images": [data],
                }
            )
        messages.append({"role": "user", "content": prompt})
        response = requests.post(
            f"{host}/api/chat",
            json={
                "model": settings.model,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {"temperature": 0.1, "num_ctx": min(settings.context_window, 16_384)},
            },
            timeout=(10, 300),
        )
        if not response.ok:
            raise RuntimeError(f"Ollama 图片分析失败（HTTP {response.status_code}）：{_ollama_error_detail(response)}")
        text = response.json().get("message", {}).get("content", "")
    else:
        from openai import OpenAI

        if settings.provider != "OpenAI 兼容（自定义）" and not settings.api_key:
            raise ValueError("请填写 API Key。")
        content = [{"type": "text", "text": prompt}]
        for label, mime_type, data in encoded:
            content.extend(
                [
                    {"type": "text", "text": label},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{data}", "detail": "high"}},
                ]
            )
        client = OpenAI(
            api_key=settings.api_key or "local-no-key-required",
            base_url=settings.base_url or None,
            timeout=300,
        )
        response = _openai_chat_with_fallback(
            client,
            {
                "model": settings.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                "temperature": 0.1,
                "max_tokens": _output_budget(settings, 1_800),
            },
        )
        text = response.choices[0].message.content

    if not text:
        raise RuntimeError("模型没有返回画面描述。请确认当前模型支持图片输入。")
    return str(text).strip()


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


def _chunk_segment_groups(
    segments: Iterable[Segment],
    max_chars: int,
    max_duration: float = 300,
) -> list[list[Segment]]:
    """Keep summary chunks bounded by both text size and timeline duration."""
    groups: list[list[Segment]] = []
    current: list[Segment] = []
    current_size = 0
    for segment in segments:
        line_size = len(f"[{format_timestamp(segment.start)}] {segment.text}") + 1
        exceeds_size = current and current_size + line_size > max_chars
        exceeds_duration = current and segment.end - current[0].start > max_duration
        if exceeds_size or exceeds_duration:
            groups.append(current)
            current = []
            current_size = 0
        current.append(segment)
        current_size += line_size
    if current:
        groups.append(current)
    return groups


def _format_segment_group(group: list[Segment]) -> str:
    return "\n".join(
        f"[{format_timestamp(segment.start)}] {segment.text}" for segment in group
    )


def _group_time_range(group: list[Segment]) -> str:
    return f"{format_timestamp(group[0].start)}–{format_timestamp(group[-1].end)}"


def _continuation_was_incomplete(text: str) -> bool:
    return "模型在自动续写时" in text and "停止续写" in text


def _without_continuation_warning(text: str) -> str:
    return text.split("\n\n> 模型在自动续写时", 1)[0].rstrip()


def _balanced_excerpt(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head_size = max_chars * 2 // 3
    tail_size = max_chars - head_size
    return f"{text[:head_size]}\n……\n{text[-tail_size:]}"


def summarize(
    transcript: Transcript,
    settings: LLMSettings,
    progress: Callable[[int, str], None] | None = None,
) -> str:
    final_output_tokens = _output_budget(settings, 3_000)
    partial_output_tokens = _output_budget(settings, 1_200)
    synthesis_output_tokens = _output_budget(settings, 1_500)
    # 为首次回答和一次自动续写预留输出空间，避免扩大输出后挤爆上下文。
    summary_input_budget = max(
        600,
        settings.context_window - final_output_tokens * 2 - 700,
    )
    chunk_size = min(6_000, summary_input_budget)
    segment_groups = _chunk_segment_groups(
        transcript.segments,
        max_chars=chunk_size,
        max_duration=300,
    )
    chunks = [_format_segment_group(group) for group in segment_groups]
    system = (
        "你是一名严谨的视频内容分析与笔记整理助手。只能依据给定字幕作答，不要虚构画面、"
        "人物、数据或事实。默认生成详细、完整的视频笔记，而不是简短概述。使用简体中文，"
        "并用 [MM:SS] 或 [HH:MM:SS] 标注章节和关键结论的依据。"
    )
    if len(chunks) == 1:
        if progress:
            progress(10, "字幕可一次处理，正在识别主题、事件和时间线……")
        result = _generate(
            settings,
            system,
            "请把以下完整字幕整理成详细视频笔记。不要只给三五句话的概括，必须覆盖视频的"
            "开头、中段和结尾，并按以下结构输出：\n\n"
            "# 视频内容详解\n"
            "## 一、主题与背景\n说明视频讨论的问题、背景和目标。\n"
            "## 二、按时间展开的详细内容\n按时间顺序划分章节；每章写明时间范围、发生或讨论了什么、"
            "重要人物/概念/数据/例子，以及该部分得出的结论。\n"
            "## 三、核心观点与论证过程\n详细说明作者如何从前提、材料或案例推导到结论，不要只列关键词。\n"
            "## 四、重要细节与可复用知识\n整理容易被简略摘要遗漏的定义、方法、数字、因果关系、"
            "对比、例外和注意事项。\n"
            "## 五、最终结论\n说明视频最后得出了什么结论，以及仍然存在的不确定信息。\n\n"
            "要求：信息密度优先，避免空话；字幕明确提到的例子和细节应尽量保留；"
            "若某项字幕没有涉及，请明确说明，不要补写。\n\n"
            f"视频标题：{transcript.title}\n\n字幕：\n{chunks[0]}",
            max_tokens=final_output_tokens,
            continuation_progress=(
                (lambda count: progress(90, f"输出较长，正在自动续写第 {count} 段……"))
                if progress
                else None
            ),
        )
        if progress:
            progress(90, "模型已完成内容分析，正在检查摘要结构……")
            progress(100, "摘要生成完成。")
        return result

    def summarize_group(group: list[Segment], label: str, depth: int = 0) -> str:
        source = _format_segment_group(group)
        time_range = _group_time_range(group)
        partial = _generate(
            settings,
            system,
            f"这是视频字幕的第 {label} 段，时间范围为 [{time_range}]。"
            "请生成详细的阶段笔记，不要只写概括，也不要写‘阶段笔记中未完整呈现’之类的占位语。"
            "必须从本段第一条字幕覆盖到最后一条字幕，按时间顺序保留事件、论点、论据、人物、"
            "概念、数字、例子、因果关系和结论。使用若干小标题或项目列表并保留时间戳；"
            "不要引入本段之外的信息。\n\n"
            f"本段字幕：\n{source}",
            max_tokens=partial_output_tokens,
        )
        if not _continuation_was_incomplete(partial):
            return partial

        # A repeated continuation means this section may have an unfinished tail.
        # Split only that source range and regenerate both halves instead of allowing
        # the final synthesis to silently replace it with a large timeline gap.
        if len(group) >= 4 and depth < 3:
            midpoint = len(group) // 2
            left_group = group[:midpoint]
            right_group = group[midpoint:]
            left = summarize_group(left_group, f"{label}A", depth + 1)
            right = summarize_group(right_group, f"{label}B", depth + 1)
            return (
                f"#### {_group_time_range(left_group)}\n{left}\n\n"
                f"#### {_group_time_range(right_group)}\n{right}"
            )

        cleaned = _without_continuation_warning(partial)
        fallback = (
            "**该小段的模型续写仍未完成，以下保留对应字幕作为覆盖保底：**\n\n"
            + source
        )
        return f"{cleaned}\n\n{fallback}" if cleaned else fallback

    partials: list[tuple[str, str]] = []
    for index, group in enumerate(segment_groups, start=1):
        if progress:
            percent = 5 + int(70 * (index - 1) / len(segment_groups))
            progress(percent, f"正在整理字幕第 {index}/{len(segment_groups)} 段……")
        partial = summarize_group(group, f"{index}/{len(segment_groups)}")
        partials.append((_group_time_range(group), partial))
        if progress:
            percent = 5 + int(70 * index / len(segment_groups))
            progress(
                percent,
                f"第 {index}/{len(segment_groups)} 段分析完成，已保留该时间范围的独立笔记。",
            )

    # The synthesis may be concise, but it is no longer allowed to replace the
    # detailed timeline. Every independently generated section is appended below
    # verbatim, so a weak merge cannot erase the middle of a long video.
    synthesis_input_budget = max(
        800,
        settings.context_window - synthesis_output_tokens * 2 - 700,
    )
    excerpt_size = max(220, synthesis_input_budget // len(partials) - 60)
    synthesis_material = "\n\n".join(
        f"[{time_range}]\n{_balanced_excerpt(partial, excerpt_size)}"
        for time_range, partial in partials
    )
    if progress:
        progress(82, "所有时间段均已保留，正在补充跨章节的主题与结论……")
    overview = _generate(
        settings,
        system,
        "请根据下面各时间段笔记生成报告的综合部分。这里只输出：\n"
        "## 一、主题与背景\n"
        "## 二、核心观点与论证过程\n"
        "## 三、重要人物、概念、数字与因果关系\n"
        "## 四、最终结论与不确定信息\n\n"
        "不要重写或压缩详细时间线；详细时间线会由程序原样附在后面。不要声称某段内容未提供，"
        "只能依据材料归纳。\n\n"
        + synthesis_material,
        max_tokens=synthesis_output_tokens,
        continuation_progress=(
            (lambda count: progress(92, f"综合分析较长，正在自动续写第 {count} 段……"))
            if progress
            else None
        ),
    )
    timeline = "\n\n".join(
        f"### {time_range}（字幕第 {index}/{len(partials)} 段）\n\n{partial}"
        for index, (time_range, partial) in enumerate(partials, start=1)
    )
    result = (
        f"# 视频内容详解\n\n{overview}\n\n"
        "## 五、按时间展开的完整内容\n\n"
        f"{timeline}"
    )
    if progress:
        progress(100, f"摘要生成完成，已保留全部 {len(partials)} 个时间段。")
    return result


def _question_terms(question: str) -> set[str]:
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", question.lower())
    terms = {normalized[index : index + 2] for index in range(max(0, len(normalized) - 1))}
    terms.update(token for token in re.findall(r"[a-z0-9_]{3,}", normalized))
    return terms


def estimate_context_tokens(transcript: Transcript, reserve_tokens: int = 2_000) -> int:
    """Conservatively estimate tokens needed for a full-transcript request."""
    text = transcript_as_text(transcript.segments)
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", text))
    other_count = max(0, len(text) - cjk_count)
    return cjk_count + (other_count + 3) // 4 + reserve_tokens


def _relevant_context(transcript: Transcript, question: str, max_chars: int = 48_000) -> str:
    full_text = transcript_as_text(transcript.segments)
    if len(full_text) <= max_chars:
        return full_text
    chunk_size = min(3_000, max(800, max_chars // 2))
    chunks = _chunk_segments(transcript.segments, max_chars=chunk_size)
    terms = _question_terms(question)

    def score(chunk: str) -> int:
        lowered = chunk.lower()
        return sum(lowered.count(term) for term in terms)

    ranked = sorted(enumerate(chunks), key=lambda pair: (score(pair[1]), -pair[0]), reverse=True)
    selected: list[tuple[int, str]] = []
    selected_size = 0
    for index, chunk in ranked:
        separator_size = 2 if selected else 0
        remaining = max_chars - selected_size - separator_size
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            if not selected:
                truncated = chunk[:remaining]
                selected.append((index, truncated))
                selected_size += len(truncated)
            continue
        selected.append((index, chunk))
        selected_size += separator_size + len(chunk)
    selected.sort(key=lambda pair: pair[0])
    return "\n\n".join(chunk for _, chunk in selected)


def answer_question(
    transcript: Transcript,
    question: str,
    settings: LLMSettings,
    history: list[dict[str, str]] | None = None,
    progress: Callable[[int, str], None] | None = None,
) -> str:
    if progress:
        progress(10, "正在分析问题并确定需要查找的字幕内容……")
    recent_history = (history or [])[-4:]
    answer_output_tokens = _output_budget(settings, 1_200)
    # 为回答和一次自动续写预留空间；其余预算用于字幕与最近对话。
    available_chars = max(
        400,
        settings.context_window - answer_output_tokens * 2 - 700,
    )
    history_limit = min(12_000, max(100, available_chars // 5))
    history_blocks: list[str] = []
    history_size = 0
    for item in reversed(recent_history):
        block = f"用户：{item.get('question', '')}\n助手：{item.get('answer', '')}"
        remaining = history_limit - history_size - (1 if history_blocks else 0)
        if remaining <= 0:
            break
        history_blocks.append(block[:remaining])
        history_size += min(len(block), remaining) + (1 if len(history_blocks) > 1 else 0)
    history_text = "\n".join(reversed(history_blocks))
    context_limit = max(250, available_chars - len(history_text))
    context = _relevant_context(transcript, question, max_chars=context_limit)
    if progress:
        progress(35, f"已从 {len(transcript.segments)} 段字幕中整理出相关依据……")
    system = (
        "你是一名严谨的视频问答助手。只依据提供的字幕回答。"
        "找不到答案时明确说字幕中没有足够信息，不要猜测。"
        "尽量在结论后引用字幕已有的 [MM:SS] 或 [HH:MM:SS] 时间戳。"
    )
    prompt = f"视频标题：{transcript.title}\n"
    if history_text:
        prompt += f"\n最近对话：\n{history_text}\n"
    prompt += f"\n问题：{question}\n\n相关字幕：\n{context}"
    if progress:
        progress(55, f"正在请求 {settings.provider} 模型根据字幕组织回答……")
    result = _generate(
        settings,
        system,
        prompt,
        max_tokens=answer_output_tokens,
        continuation_progress=(
            (lambda count: progress(85, f"回答较长，正在自动续写第 {count} 段……"))
            if progress
            else None
        ),
    )
    if progress:
        progress(100, "回答生成完成，正在显示结果。")
    return result
