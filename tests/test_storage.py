from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from bili_reader.models import Segment, Transcript
from bili_reader.storage import (
    add_history_item,
    consolidate_question_history_items,
    delete_history_item,
    list_deleted_history,
    list_history,
    load_app_settings,
    purge_expired_history,
    restore_deleted_history_item,
    save_transcript,
    save_app_settings,
    save_video_history_item,
    soft_delete_history_item,
    update_history_item,
)


class StorageTests(unittest.TestCase):
    def test_app_settings_are_saved_locally_and_loaded_after_restart(self) -> None:
        payload = {
            "remember_api_key": True,
            "llm": {"provider": "Ollama", "model": "vision-model", "context_window": 8192},
            "reader": {"whisper_model": "small", "auth_mode": "不使用"},
        }
        with tempfile.TemporaryDirectory() as temporary, patch(
            "bili_reader.storage.DATA_ROOT", Path(temporary)
        ):
            saved_path = save_app_settings(payload)
            self.assertEqual(saved_path, Path(temporary) / "settings.json")
            self.assertEqual(load_app_settings(), payload)

    def test_history_create_archive_and_delete(self) -> None:
        transcript = Transcript(
            "video-id",
            "测试视频",
            "test",
            "zh",
            [Segment(0, 10, "字幕内容")],
        )
        with tempfile.TemporaryDirectory() as temporary, patch(
            "bili_reader.storage.DATA_ROOT", Path(temporary)
        ):
            item = add_history_item(
                transcript=transcript,
                kind="summary",
                title="详细总结",
                content="完整内容",
                provider="Ollama",
                model="test-model",
                processing_seconds=12.5,
            )
            active = list_history()
            self.assertEqual(len(active), 1)
            self.assertEqual(active[0]["content"], "完整内容")
            self.assertEqual(active[0]["processing_seconds"], 12.5)

            update_history_item(item["id"], archived=True)
            self.assertEqual(list_history(), [])
            archived = list_history(include_archived=True)
            self.assertTrue(archived[0]["archived"])

            self.assertTrue(delete_history_item(item["id"]))
            self.assertEqual(list_history(include_archived=True), [])

    def test_questions_are_saved_inside_one_video_conversation(self) -> None:
        transcript = Transcript(
            "video-id",
            "测试视频",
            "test",
            "zh",
            [Segment(0, 10, "字幕内容")],
        )
        chat = [{"question": "问题", "answer": "回答"}]
        with tempfile.TemporaryDirectory() as temporary, patch(
            "bili_reader.storage.DATA_ROOT", Path(temporary)
        ):
            first = save_video_history_item(
                transcript=transcript,
                provider="OpenAI",
                model="test-model",
                chat_history=chat,
            )
            save_video_history_item(
                transcript=transcript,
                provider="OpenAI",
                model="test-model",
                chat_history=chat + [{"question": "问题2", "answer": "回答2"}],
                preferred_item_id=first["id"],
            )
            history = list_history()
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["kind"], "video")
            self.assertEqual(history[0]["title"], "测试视频")
            self.assertEqual(len(history[0]["chat_history"]), 2)

    def test_edited_question_branches_replace_visible_tail_without_losing_old_branch(self) -> None:
        transcript = Transcript(
            "video-id",
            "测试视频",
            "test",
            "zh",
            [Segment(0, 10, "字幕内容")],
        )
        original = [
            {"question": "原问题", "answer": "原回答"},
            {"question": "旧后续", "answer": "旧后续回答"},
        ]
        edited = [{"question": "修改后的问题", "answer": "新回答"}]
        branches = [
            {"id": "original", "history": original, "fork_index": 0},
            {"id": "edited", "history": edited, "fork_index": 0},
        ]
        with tempfile.TemporaryDirectory() as temporary, patch(
            "bili_reader.storage.DATA_ROOT", Path(temporary)
        ):
            item = save_video_history_item(
                transcript=transcript,
                provider="OpenAI",
                model="test-model",
                chat_history=original,
            )
            save_video_history_item(
                transcript=transcript,
                provider="OpenAI",
                model="test-model",
                chat_history=edited,
                chat_branches=branches,
                active_chat_branch="edited",
                preferred_item_id=item["id"],
            )
            saved = list_history()[0]
            self.assertEqual(saved["chat_history"], edited)
            self.assertEqual(saved["chat_branches"], branches)
            self.assertEqual(saved["active_chat_branch"], "edited")

    def test_legacy_question_entries_are_folded_into_the_video(self) -> None:
        transcript = Transcript(
            "video-id",
            "测试视频",
            "test",
            "zh",
            [Segment(0, 10, "字幕内容")],
        )
        with tempfile.TemporaryDirectory() as temporary, patch(
            "bili_reader.storage.DATA_ROOT", Path(temporary)
        ):
            summary = add_history_item(
                transcript=transcript,
                kind="summary",
                title="详细总结",
                content="总结",
                provider="Ollama",
                model="model-a",
            )
            add_history_item(
                transcript=transcript,
                kind="qa",
                title="问题",
                content="回答",
                provider="Ollama",
                model="model-a",
                chat_history=[{"question": "问题", "answer": "回答"}],
            )
            self.assertEqual(consolidate_question_history_items(), 1)
            history = list_history()
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["id"], summary["id"])
            self.assertEqual(history[0]["kind"], "video")
            self.assertEqual(history[0]["chat_history"][0]["question"], "问题")

    def test_soft_delete_moves_content_to_15_day_backup_and_restores_it(self) -> None:
        transcript = Transcript(
            "video-id",
            "测试视频",
            "test",
            "zh",
            [Segment(0, 10, "字幕内容")],
        )
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temporary, patch(
            "bili_reader.storage.DATA_ROOT", Path(temporary)
        ):
            save_transcript(transcript)
            video_path = Path(temporary) / "video-id"
            (video_path / "video-id-audio.mp3").write_bytes(b"audio")
            item = add_history_item(
                transcript=transcript,
                kind="summary",
                title="详细总结",
                content="完整内容",
                provider="Ollama",
                model="test-model",
            )

            deleted = soft_delete_history_item(item["id"], now=now)
            self.assertIsNotNone(deleted)
            self.assertFalse(video_path.exists())
            self.assertEqual(list_history(), [])
            trash = list_deleted_history(now=now)
            self.assertEqual(len(trash), 1)
            expires_at = datetime.fromisoformat(trash[0]["expires_at"])
            self.assertEqual(expires_at, now + timedelta(days=15))
            backup = Path(temporary) / ".trash" / item["id"] / "content"
            self.assertTrue((backup / "transcript.json").exists())
            self.assertTrue((backup / "video-id-audio.mp3").exists())

            restored = restore_deleted_history_item(item["id"])
            self.assertIsNotNone(restored)
            self.assertTrue((video_path / "transcript.json").exists())
            self.assertTrue((video_path / "video-id-audio.mp3").exists())
            self.assertEqual(len(list_history()), 1)
            self.assertEqual(list_deleted_history(now=now), [])

    def test_soft_delete_survives_a_windows_locked_file(self) -> None:
        transcript = Transcript(
            "locked-video",
            "被占用的视频",
            "test",
            "zh",
            [Segment(0, 10, "字幕内容")],
        )
        with tempfile.TemporaryDirectory() as temporary, patch(
            "bili_reader.storage.DATA_ROOT", Path(temporary)
        ):
            save_transcript(transcript)
            video_path = Path(temporary) / "locked-video"
            (video_path / "debug.log").write_text("busy", encoding="utf-8")
            item = add_history_item(
                transcript=transcript,
                kind="video",
                title=transcript.title,
                content="内容",
                provider="Ollama",
                model="test-model",
            )
            real_rmtree = __import__("shutil").rmtree

            def locked_rmtree(path, *args, **kwargs):
                if Path(path) == video_path:
                    raise PermissionError(32, "文件正在使用", str(video_path / "debug.log"))
                return real_rmtree(path, *args, **kwargs)

            with patch("bili_reader.storage.shutil.rmtree", side_effect=locked_rmtree):
                deleted = soft_delete_history_item(item["id"])
                trash = list_deleted_history()

            self.assertTrue(deleted["content_cleanup_pending"])
            self.assertTrue(video_path.exists())
            self.assertTrue(trash[0]["content_backup"])
            self.assertTrue(
                (
                    Path(temporary)
                    / ".trash"
                    / item["id"]
                    / "content"
                    / "transcript.json"
                ).exists()
            )

            # A later page load retries after the writer releases the file.
            refreshed = list_deleted_history()
            self.assertFalse(video_path.exists())
            self.assertNotIn("content_cleanup_pending", refreshed[0])

    def test_expired_backup_is_permanently_purged(self) -> None:
        transcript = Transcript(
            "video-id",
            "测试视频",
            "test",
            "zh",
            [Segment(0, 10, "字幕内容")],
        )
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temporary, patch(
            "bili_reader.storage.DATA_ROOT", Path(temporary)
        ):
            save_transcript(transcript)
            item = add_history_item(
                transcript=transcript,
                kind="qa",
                title="问题",
                content="回答",
                provider="Ollama",
                model="test-model",
            )
            soft_delete_history_item(item["id"], now=now)
            self.assertEqual(
                purge_expired_history(now=now + timedelta(days=16)),
                1,
            )
            self.assertEqual(list_deleted_history(now=now + timedelta(days=16)), [])
            self.assertFalse((Path(temporary) / ".trash" / item["id"]).exists())

    def test_shared_video_content_stays_until_last_active_history_is_deleted(self) -> None:
        transcript = Transcript(
            "video-id",
            "测试视频",
            "test",
            "zh",
            [Segment(0, 10, "字幕内容")],
        )
        with tempfile.TemporaryDirectory() as temporary, patch(
            "bili_reader.storage.DATA_ROOT", Path(temporary)
        ):
            save_transcript(transcript)
            first = add_history_item(
                transcript=transcript,
                kind="summary",
                title="总结",
                content="总结内容",
                provider="Ollama",
                model="test-model",
            )
            second = add_history_item(
                transcript=transcript,
                kind="qa",
                title="问题",
                content="回答",
                provider="Ollama",
                model="test-model",
            )
            video_path = Path(temporary) / "video-id"

            soft_delete_history_item(first["id"])
            self.assertTrue(video_path.exists())
            soft_delete_history_item(second["id"])
            self.assertFalse(video_path.exists())
            self.assertEqual(len(list_deleted_history()), 2)


if __name__ == "__main__":
    unittest.main()
