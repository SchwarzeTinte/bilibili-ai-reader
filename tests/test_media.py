from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from bili_reader.extractor import _has_audio_stream
from bili_reader.llm import LLMSettings, vision_support_status
from bili_reader.visual import frame_sampling_plan, vision_batch_size


class MediaFallbackTests(unittest.TestCase):
    def test_audio_stream_detection_uses_format_metadata(self) -> None:
        self.assertTrue(_has_audio_stream({"formats": [{"acodec": "mp4a.40.2"}]}))
        self.assertFalse(_has_audio_stream({"formats": [{"acodec": "none"}]}))
        self.assertIsNone(_has_audio_stream({"formats": []}))

    def test_frame_sampling_is_adaptive_and_bounded(self) -> None:
        short_count, _ = frame_sampling_plan(15 * 60)
        long_count, _ = frame_sampling_plan(4 * 60 * 60)
        self.assertEqual(short_count, 60)
        self.assertEqual(long_count, 180)

    def test_local_vision_uses_single_frame_requests(self) -> None:
        local = LLMSettings("Ollama", "vision-model", "", "http://localhost:11434/v1")
        cloud = LLMSettings("Gemini", "vision-model", "key", "")
        self.assertEqual(vision_batch_size(local), 1)
        self.assertEqual(vision_batch_size(cloud), 3)

    def test_ollama_vision_capability_is_detected(self) -> None:
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"capabilities": ["completion", "vision"]},
        )
        settings = LLMSettings("Ollama", "local-vision", "", "http://localhost:11434/v1")
        with patch("bili_reader.llm.requests.post", return_value=response):
            status, message = vision_support_status(settings)
        self.assertEqual(status, "supported")
        self.assertIn("vision", message)

    def test_plain_deepseek_model_is_marked_as_text_only(self) -> None:
        status, _ = vision_support_status(
            LLMSettings("DeepSeek", "deepseek-chat", "key", "https://api.deepseek.com")
        )
        self.assertEqual(status, "unsupported")


if __name__ == "__main__":
    unittest.main()
