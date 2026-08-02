from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

from bili_reader.llm import (
    LLMSettings,
    _generate,
    _ollama_error_detail,
    _relevant_context,
    answer_question,
    describe_images,
    estimate_context_tokens,
    summarize,
    test_connection,
)
from bili_reader.models import Segment, Transcript


def long_transcript(segment_count: int = 500) -> Transcript:
    return Transcript(
        video_id="test-video",
        title="测试视频",
        source="test",
        language="zh",
        segments=[
            Segment(index * 2, index * 2 + 2, f"第{index}段字幕包含用于测试的详细内容。")
            for index in range(segment_count)
        ],
    )


class LLMTests(unittest.TestCase):
    def test_ollama_vision_sends_local_frame_as_base64(self) -> None:
        settings = LLMSettings("Ollama", "vision-model", "", "http://localhost:11434/v1")
        response = SimpleNamespace(
            ok=True,
            status_code=200,
            json=lambda: {"message": {"content": "[00:00] 画面中有一张图表"}},
        )
        with tempfile.TemporaryDirectory() as temporary:
            frame = Path(temporary) / "frame.jpg"
            frame.write_bytes(b"fake-jpeg")
            with patch("bili_reader.llm.requests.post", return_value=response) as post:
                result = describe_images(settings, [("[00:00]", frame)])
        self.assertIn("图表", result)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertTrue(payload["messages"][1]["images"][0])
        self.assertIn("[00:00]", payload["messages"][1]["content"])
        self.assertFalse(payload["stream"])

    def test_relevant_context_never_exceeds_budget(self) -> None:
        context = _relevant_context(long_transcript(), "详细内容", max_chars=5_000)
        self.assertLessEqual(len(context), 5_000)
        self.assertTrue(context)

        oversized_segment = Transcript(
            "large-segment",
            "长字幕",
            "test",
            "zh",
            [Segment(0, 1, "字" * 10_000)],
        )
        context = _relevant_context(oversized_segment, "字幕", max_chars=1_000)
        self.assertLessEqual(len(context), 1_000)

    def test_question_limits_context_and_history_and_reports_progress(self) -> None:
        transcript = long_transcript()
        history = [
            {"question": f"历史问题 {index}", "answer": "历史回答" * 1_000}
            for index in range(4)
        ]
        events: list[tuple[int, str]] = []
        settings = LLMSettings("Ollama", "test-model", "", "http://localhost:11434/v1")
        with patch("bili_reader.llm._generate", return_value="回答成功") as generate:
            result = answer_question(
                transcript,
                "请详细说明",
                settings,
                history,
                progress=lambda percent, message: events.append((percent, message)),
            )

        self.assertEqual(result, "回答成功")
        prompt = generate.call_args.args[2]
        self.assertLess(len(prompt), 7_000)
        self.assertEqual([percent for percent, _ in events], [10, 35, 55, 100])

    def test_larger_context_can_include_the_full_transcript(self) -> None:
        transcript = long_transcript()
        settings = LLMSettings(
            "Ollama",
            "test-model",
            "",
            "http://localhost:11434/v1",
            context_window=32_768,
        )
        with patch("bili_reader.llm._generate", return_value="回答成功") as generate:
            answer_question(transcript, "请介绍全部内容", settings)
        prompt = generate.call_args.args[2]
        self.assertIn("第499段字幕包含用于测试的详细内容", prompt)
        self.assertGreater(estimate_context_tokens(transcript), 2_000)

    def test_ollama_request_uses_smallest_sufficient_context_window(self) -> None:
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"message": {"content": "回答成功"}}).encode("utf-8")
        settings = LLMSettings(
            "Ollama",
            "test-model",
            "",
            "http://localhost:11434/v1",
            context_window=16_384,
        )
        with patch("bili_reader.llm.requests.post", return_value=response) as post:
            self.assertEqual(_generate(settings, "system", "prompt"), "回答成功")
        self.assertEqual(post.call_args.kwargs["json"]["options"]["num_ctx"], 2_048)
        self.assertTrue(post.call_args.kwargs["stream"])
        self.assertEqual(post.call_args.kwargs["timeout"], (10, None))

    def test_ollama_runtime_context_grows_with_the_actual_prompt(self) -> None:
        response = requests.Response()
        response.status_code = 200
        response._content = (
            json.dumps({"message": {"content": "回答"}})
            + "\n"
            + json.dumps(
                {"message": {"content": "成功"}, "done": True, "done_reason": "stop"}
            )
        ).encode("utf-8")
        settings = LLMSettings(
            "Ollama",
            "test-model",
            "",
            "http://localhost:11434/v1",
            context_window=16_384,
        )
        with patch("bili_reader.llm.requests.post", return_value=response) as post:
            self.assertEqual(_generate(settings, "system", "字" * 6_000), "回答成功")
        self.assertEqual(post.call_args.kwargs["json"]["options"]["num_ctx"], 8_192)

    def test_ollama_automatically_continues_when_output_hits_length_limit(self) -> None:
        responses = []
        for content, reason in (("第一部分尚未完成", "length"), ("第二部分已经完成。", "stop")):
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps(
                {"message": {"content": content}, "done_reason": reason}
            ).encode("utf-8")
            responses.append(response)
        settings = LLMSettings(
            "Ollama",
            "test-model",
            "",
            "http://localhost:11434/v1",
            context_window=8_192,
        )
        with patch("bili_reader.llm.requests.post", side_effect=responses) as post:
            result = _generate(settings, "system", "prompt", max_tokens=1_000)
        self.assertIn("第一部分尚未完成", result)
        self.assertIn("第二部分已经完成", result)
        self.assertEqual(post.call_count, 2)
        continuation = post.call_args_list[1].kwargs["json"]["messages"][1]["content"]
        self.assertIn("达到输出长度上限", continuation)

    def test_summary_reports_segment_analysis_steps(self) -> None:
        events: list[tuple[int, str]] = []
        settings = LLMSettings("Ollama", "test-model", "", "http://localhost:11434/v1")
        with patch("bili_reader.llm._generate", return_value="阶段结果"):
            summarize(
                long_transcript(),
                settings,
                progress=lambda percent, message: events.append((percent, message)),
            )
        self.assertTrue(any("分析完成" in message for _, message in events))
        self.assertEqual(events[-1][0], 100)

    def test_nested_ollama_error_is_readable(self) -> None:
        response = requests.Response()
        response.status_code = 400
        nested = {"error": {"message": "request exceeds the available context size"}}
        response._content = json.dumps({"error": json.dumps(nested)}).encode("utf-8")
        self.assertEqual(
            _ollama_error_detail(response),
            "request exceeds the available context size",
        )

    def test_openai_compatible_providers_use_expected_options(self) -> None:
        cases = [
            ("OpenAI", "gpt-4.1-mini", "https://api.openai.com/v1", False),
            ("DeepSeek", "deepseek-v4-flash", "https://api.deepseek.com", True),
        ]
        for provider, model, base_url, disables_thinking in cases:
            with self.subTest(provider=provider), patch("openai.OpenAI") as client_class:
                client_class.return_value.chat.completions.create.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="连接成功"))]
                )
                result = test_connection(LLMSettings(provider, model, "test-key", base_url))
                options = client_class.return_value.chat.completions.create.call_args.kwargs
                self.assertEqual(result, "连接成功")
                self.assertEqual(options["model"], model)
                self.assertEqual("extra_body" in options, disables_thinking)

    def test_custom_openai_compatible_service_can_work_without_api_key(self) -> None:
        with patch("openai.OpenAI") as client_class:
            client_class.return_value.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="连接成功"))]
            )
            result = test_connection(
                LLMSettings(
                    "OpenAI 兼容（自定义）",
                    "local-model",
                    "",
                    "http://localhost:1234/v1",
                )
            )
        self.assertEqual(result, "连接成功")
        self.assertEqual(client_class.call_args.kwargs["api_key"], "local-no-key-required")
        self.assertEqual(
            client_class.call_args.kwargs["base_url"],
            "http://localhost:1234/v1",
        )

    def test_openai_compatible_falls_back_for_common_parameter_differences(self) -> None:
        attempts: list[dict[str, object]] = []

        def create(**options):
            attempts.append(options)
            if len(attempts) == 1:
                raise RuntimeError("temperature is not supported")
            if len(attempts) == 2:
                raise RuntimeError("max_tokens is not supported")
            if len(attempts) == 3:
                raise RuntimeError("system role is not supported")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="兼容成功"))]
            )

        with patch("openai.OpenAI") as client_class:
            client_class.return_value.chat.completions.create.side_effect = create
            result = _generate(
                LLMSettings(
                    "OpenAI 兼容（自定义）",
                    "different-server-model",
                    "",
                    "http://localhost:9000/v1",
                ),
                "system prompt",
                "user prompt",
            )

        self.assertEqual(result, "兼容成功")
        self.assertNotIn("temperature", attempts[-1])
        self.assertNotIn("max_tokens", attempts[-1])
        self.assertIn("max_completion_tokens", attempts[-1])
        self.assertEqual(len(attempts[-1]["messages"]), 1)

    def test_openai_compatible_automatically_continues_length_finish(self) -> None:
        responses = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="第一部分"),
                        finish_reason="length",
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="第二部分"),
                        finish_reason="stop",
                    )
                ]
            ),
        ]
        with patch("openai.OpenAI") as client_class:
            client_class.return_value.chat.completions.create.side_effect = responses
            result = _generate(
                LLMSettings(
                    "OpenAI 兼容（自定义）",
                    "local-model",
                    "",
                    "http://localhost:1234/v1",
                ),
                "system",
                "prompt",
                max_tokens=1_000,
            )
            calls = client_class.return_value.chat.completions.create.call_args_list
        self.assertEqual(result, "第一部分\n第二部分")
        self.assertEqual(len(calls), 2)
        self.assertIn("达到输出长度上限", calls[1].kwargs["messages"][1]["content"])

    def test_generation_continues_more_than_two_times_until_normal_stop(self) -> None:
        responses = [
            SimpleNamespace(text="第一段", truncated=True),
            SimpleNamespace(text="第二段", truncated=True),
            SimpleNamespace(text="第三段", truncated=True),
            SimpleNamespace(text="第四段完成。", truncated=False),
        ]
        progress: list[int] = []
        with patch("bili_reader.llm._generate_once", side_effect=responses) as generate_once:
            result = _generate(
                LLMSettings("Ollama", "test-model", context_window=8_192),
                "system",
                "prompt",
                max_tokens=1_000,
                continuation_progress=progress.append,
            )
        self.assertEqual(generate_once.call_count, 4)
        self.assertEqual(progress, [1, 2, 3])
        self.assertIn("第四段完成", result)

    def test_generation_stops_when_continuation_repeats(self) -> None:
        responses = [
            SimpleNamespace(text="已经生成的内容", truncated=True),
            SimpleNamespace(text="已经生成的内容", truncated=True),
        ]
        with patch("bili_reader.llm._generate_once", side_effect=responses) as generate_once:
            result = _generate(
                LLMSettings("Ollama", "test-model", context_window=8_192),
                "system",
                "prompt",
                max_tokens=1_000,
            )
        self.assertEqual(generate_once.call_count, 2)
        self.assertIn("避免无限循环", result)

    def test_anthropic_provider_uses_messages_api(self) -> None:
        with patch("anthropic.Anthropic") as client_class:
            client_class.return_value.messages.create.return_value = SimpleNamespace(
                content=[SimpleNamespace(type="text", text="连接成功")]
            )
            result = test_connection(
                LLMSettings("Anthropic", "claude-sonnet-5", "test-key")
            )
            options = client_class.return_value.messages.create.call_args.kwargs
        self.assertEqual(result, "连接成功")
        self.assertEqual(options["model"], "claude-sonnet-5")
        self.assertEqual(options["messages"][0]["role"], "user")

    def test_summary_chunk_size_follows_context_budget(self) -> None:
        settings = LLMSettings(
            "OpenAI 兼容（自定义）",
            "small-context-model",
            "",
            "http://localhost:1234/v1",
            context_window=4_096,
        )
        with patch("bili_reader.llm._generate", return_value="阶段结果") as generate:
            summarize(long_transcript(), settings)
        first_chunk_prompt = generate.call_args_list[0].args[2]
        self.assertLess(len(first_chunk_prompt), 4_096)
        self.assertGreater(generate.call_count, 2)

    def test_summary_uses_adaptive_larger_final_output_budget(self) -> None:
        transcript = Transcript(
            "short-video",
            "短视频",
            "test",
            "zh",
            [Segment(0, 10, "这是足够短、可以一次处理的字幕。")],
        )
        settings = LLMSettings(
            "OpenAI 兼容（自定义）",
            "local-model",
            "",
            "http://localhost:1234/v1",
            context_window=8_192,
        )
        with patch("bili_reader.llm._generate", return_value="完整总结") as generate:
            result = summarize(transcript, settings)
        self.assertEqual(result, "完整总结")
        self.assertEqual(generate.call_args.kwargs["max_tokens"], 2_048)
        self.assertIn("详细视频笔记", generate.call_args.args[2])

    def test_summary_preserves_every_generated_timeline_section(self) -> None:
        settings = LLMSettings(
            "OpenAI 兼容（自定义）",
            "small-context-model",
            "",
            "http://localhost:1234/v1",
            context_window=4_096,
        )
        section_count = 0
        section_outputs: list[str] = []

        def generate(_settings, _system, prompt, **_kwargs):
            nonlocal section_count
            if "本段字幕" in prompt:
                section_count += 1
                output = f"UNIQUE_TIMELINE_SECTION_{section_count}"
                section_outputs.append(output)
                return output
            return "综合结论"

        with patch("bili_reader.llm._generate", side_effect=generate):
            result = summarize(long_transcript(), settings)

        self.assertGreater(section_count, 2)
        self.assertIn("按时间展开的完整内容", result)
        for output in section_outputs:
            self.assertIn(output, result)

    def test_summary_splits_only_the_section_with_repeating_continuation(self) -> None:
        settings = LLMSettings(
            "OpenAI 兼容（自定义）",
            "small-context-model",
            "",
            "http://localhost:1234/v1",
            context_window=4_096,
        )
        returned_incomplete = False
        recovered_sections = 0

        def generate(_settings, _system, prompt, **_kwargs):
            nonlocal returned_incomplete, recovered_sections
            if "本段字幕" not in prompt:
                return "综合结论"
            if not returned_incomplete:
                returned_incomplete = True
                return (
                    "尚未完成\n\n> 模型在自动续写时开始重复、没有产生新内容，"
                    "程序已停止续写以避免无限循环。"
                )
            recovered_sections += 1
            return f"恢复后的子段 {recovered_sections}"

        with patch("bili_reader.llm._generate", side_effect=generate):
            result = summarize(long_transcript(), settings)

        self.assertGreaterEqual(recovered_sections, 2)
        self.assertNotIn("程序已停止续写以避免无限循环", result)
        self.assertIn("恢复后的子段", result)

    def test_gemini_provider_uses_selected_model(self) -> None:
        with patch("google.genai.Client") as client_class:
            client_class.return_value.models.generate_content.return_value = SimpleNamespace(
                text="连接成功"
            )
            result = test_connection(LLMSettings("Gemini", "gemini-2.5-flash", "test-key"))
            options = client_class.return_value.models.generate_content.call_args.kwargs
        self.assertEqual(result, "连接成功")
        self.assertEqual(options["model"], "gemini-2.5-flash")


if __name__ == "__main__":
    unittest.main()
