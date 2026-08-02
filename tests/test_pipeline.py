from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from bili_reader.llm import LLMSettings
from bili_reader.models import Segment, Transcript
from bili_reader.pipeline import (
    merge_text_and_visual,
    smart_read_video,
    transcript_text_quality,
)


class SmartPipelineTests(unittest.TestCase):
    def test_text_density_threshold_detects_too_little_content(self) -> None:
        sparse = Transcript("v", "视频", "字幕", "zh", [Segment(0, 900, "字" * 200)])
        rich = Transcript("v", "视频", "字幕", "zh", [Segment(0, 900, "字" * 900)])
        self.assertFalse(transcript_text_quality(sparse, 900).sufficient)
        self.assertTrue(transcript_text_quality(rich, 900).sufficient)

    def test_timeline_coverage_also_triggers_visual_fallback(self) -> None:
        intro_only = Transcript(
            "v",
            "视频",
            "字幕",
            "zh",
            [Segment(0, 60, "字" * 1_000)],
        )
        quality = transcript_text_quality(intro_only, 900)
        self.assertFalse(quality.sufficient)
        self.assertLess(quality.timeline_coverage, 0.35)

    def test_merge_preserves_text_and_visual_segments(self) -> None:
        text = Transcript("v", "视频", "Whisper", "zh", [Segment(0, 10, "语音内容")])
        visual = Transcript("v", "视频", "画面", "visual", [Segment(5, 15, "画面内容")])
        combined = merge_text_and_visual(text, visual)
        self.assertEqual([segment.text for segment in combined.segments], ["语音内容", "画面内容"])
        self.assertIn("Whisper", combined.source)
        self.assertIn("画面", combined.source)

    def test_smart_read_adds_visual_when_subtitle_and_audio_are_both_sparse(self) -> None:
        part = {
            "id": "v",
            "title": "视频",
            "duration": 900,
            "tracks": [{"name": "字幕"}],
            "has_audio": True,
        }
        subtitle = Transcript("v", "视频", "字幕", "zh", [Segment(0, 30, "字" * 50)])
        audio = Transcript("v", "视频", "Whisper", "zh", [Segment(0, 100, "字" * 100)])
        visual = Transcript("v", "视频", "画面", "visual", [Segment(0, 900, "完整画面")])
        settings = LLMSettings("Gemini", "gemini-test", "key")
        with (
            patch("bili_reader.pipeline.download_subtitle", return_value=subtitle),
            patch("bili_reader.pipeline.download_audio", return_value=Path("audio.mp3")),
            patch("bili_reader.pipeline.transcribe_audio", return_value=audio),
            patch("bili_reader.pipeline.download_video", return_value=Path("video.mp4")),
            patch("bili_reader.pipeline.analyze_video_frames", return_value=visual),
        ):
            result = smart_read_video(
                part,
                auth_source=None,
                whisper_model="small",
                whisper_language="zh",
                whisper_device="cpu",
                settings=settings,
                progress=lambda *_: None,
            )
        self.assertIn("完整画面", [segment.text for segment in result.transcript.segments])
        self.assertIn("字" * 100, [segment.text for segment in result.transcript.segments])
        self.assertIn("自动合并", result.message)
        self.assertTrue(result.used_visual)


if __name__ == "__main__":
    unittest.main()
