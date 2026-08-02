from __future__ import annotations

import threading
import time
import unittest
import subprocess
import sys

from bili_reader.jobs import BackgroundJobManager


class BackgroundJobTests(unittest.TestCase):
    def test_running_worker_does_not_keep_a_stopped_app_process_alive(self) -> None:
        script = """
import threading
from bili_reader.jobs import BackgroundJobManager
started = threading.Event()
release = threading.Event()
def work(progress):
    started.set()
    release.wait(30)
manager = BackgroundJobManager(max_workers=1)
manager.submit(work, 'starting')
assert started.wait(2)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=".",
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_job_continues_independently_of_page_reruns(self) -> None:
        manager = BackgroundJobManager(max_workers=2)
        release = threading.Event()
        started = threading.Event()

        def work(progress):
            progress(25, "模型正在运行")
            started.set()
            release.wait(timeout=5)
            progress(90, "正在整理结果")
            return "完成内容"

        job_id = manager.submit(work, "等待开始")
        self.assertTrue(started.wait(timeout=2))
        running = manager.snapshot(job_id)
        self.assertIsNotNone(running)
        self.assertEqual(running.status, "running")
        self.assertEqual(running.progress, 25)

        # A Streamlit rerun only reads the snapshot; it does not own or cancel the worker.
        for _ in range(3):
            self.assertEqual(manager.snapshot(job_id).status, "running")
        release.set()
        for _ in range(100):
            completed = manager.snapshot(job_id)
            if completed and completed.status == "completed":
                break
            time.sleep(0.01)
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.result, "完成内容")

    def test_job_snapshot_keeps_estimate_and_lifecycle_times(self) -> None:
        manager = BackgroundJobManager(max_workers=1)
        release = threading.Event()
        started = threading.Event()

        def work(progress):
            started.set()
            release.wait(timeout=5)
            return "ok"

        job_id = manager.submit(work, "等待开始")
        manager.set_estimate(job_id, 185, "按输入长度粗略估算")
        self.assertTrue(started.wait(timeout=2))
        running = manager.snapshot(job_id)
        self.assertGreater(running.created_at, 0)
        self.assertIsNotNone(running.started_at)
        self.assertIsNone(running.finished_at)
        self.assertEqual(running.estimated_seconds, 185)
        self.assertEqual(running.estimate_note, "按输入长度粗略估算")

        release.set()
        for _ in range(100):
            completed = manager.snapshot(job_id)
            if completed and completed.status == "completed":
                break
            time.sleep(0.01)
        self.assertIsNotNone(completed.finished_at)
        self.assertGreaterEqual(completed.finished_at, completed.started_at)

    def test_multiple_jobs_can_run_concurrently(self) -> None:
        manager = BackgroundJobManager(max_workers=2)
        release = threading.Event()
        both_started = threading.Barrier(3)

        def work(progress):
            both_started.wait(timeout=2)
            release.wait(timeout=5)
            return "ok"

        first = manager.submit(work, "任务一")
        second = manager.submit(work, "任务二")
        both_started.wait(timeout=2)
        self.assertEqual(manager.snapshot(first).status, "running")
        self.assertEqual(manager.snapshot(second).status, "running")
        release.set()

    def test_metadata_can_restore_job_routing_after_a_page_session_changes(self) -> None:
        manager = BackgroundJobManager(max_workers=1)
        release = threading.Event()
        started = threading.Event()

        def work(progress):
            progress(30, "正在处理")
            started.set()
            release.wait(timeout=5)
            return "ok"

        job_id = manager.submit(work, "等待开始")
        manager.set_metadata(
            job_id,
            {"job_group": "ai", "workspace_id": "conversation-1", "kind": "summary"},
        )
        self.assertTrue(started.wait(timeout=2))
        snapshots = manager.snapshots()
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].metadata["workspace_id"], "conversation-1")
        release.set()

    def test_cancel_discards_a_running_job_result(self) -> None:
        manager = BackgroundJobManager(max_workers=1)
        release = threading.Event()
        started = threading.Event()

        def work(progress):
            progress(20, "模型正在运行")
            started.set()
            release.wait(timeout=5)
            return "不应保存的结果"

        job_id = manager.submit(work, "等待开始")
        self.assertTrue(started.wait(timeout=2))
        self.assertTrue(manager.cancel(job_id))
        self.assertEqual(manager.snapshot(job_id).status, "canceling")
        self.assertFalse(manager.cancel(job_id))
        release.set()
        for _ in range(100):
            snapshot = manager.snapshot(job_id)
            if snapshot and snapshot.status == "canceled":
                break
            time.sleep(0.01)
        self.assertEqual(snapshot.status, "canceled")
        self.assertIsNone(snapshot.result)


if __name__ == "__main__":
    unittest.main()
