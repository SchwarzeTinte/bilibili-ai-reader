from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from queue import Queue
from threading import Lock, Thread, local
from time import time
from typing import Any, Callable
from uuid import uuid4


ProgressCallback = Callable[[int, str], None]
JobFunction = Callable[[ProgressCallback], Any]
_job_context = local()


@dataclass(slots=True)
class JobSnapshot:
    job_id: str
    status: str
    progress: int
    message: str
    result: Any = None
    error: Exception | None = None
    metadata: dict[str, Any] | None = None
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    cancel_requested_at: float | None = None
    estimated_seconds: float | None = None
    estimate_note: str = ""


class JobCancelledError(RuntimeError):
    """Raised cooperatively when the user cancels a background job."""


class BackgroundJobManager:
    """Run model calls outside Streamlit's rerunnable script thread."""

    def __init__(self, max_workers: int = 4) -> None:
        self._lock = Lock()
        self._jobs: dict[str, JobSnapshot] = {}
        self._cancel_requested: set[str] = set()
        self._queue: Queue[tuple[str, JobFunction] | None] = Queue()
        self._workers = [
            Thread(
                target=self._worker_loop,
                name=f"bilibili-ai-{index + 1}",
                daemon=True,
            )
            for index in range(max_workers)
        ]
        for worker in self._workers:
            worker.start()

    def submit(self, function: JobFunction, initial_message: str) -> str:
        job_id = uuid4().hex
        with self._lock:
            self._jobs[job_id] = JobSnapshot(
                job_id=job_id,
                status="queued",
                progress=0,
                message=initial_message,
                created_at=time(),
            )

        self._queue.put((job_id, function))
        return job_id

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._execute(*item)
            finally:
                self._queue.task_done()

    def _execute(self, job_id: str, function: JobFunction) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job_id in self._cancel_requested:
                job.status = "canceled"
                job.message = "任务已终止。"
                job.finished_at = time()
                return
            job.status = "running"
            job.started_at = time()

        def update(percent: int, message: str) -> None:
            with self._lock:
                if job_id in self._cancel_requested:
                    raise JobCancelledError("任务已由用户终止。")
                job = self._jobs.get(job_id)
                if job is None:
                    raise JobCancelledError("任务已不存在。")
                job.progress = max(0, min(100, int(percent)))
                job.message = str(message)

        try:
            _job_context.manager = self
            _job_context.job_id = job_id
            result = function(update)
        except JobCancelledError:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job.status = "canceled"
                    job.message = "任务已终止。"
                    job.finished_at = time()
            return
        except Exception as exc:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                if job_id in self._cancel_requested:
                    job.status = "canceled"
                    job.message = "任务已终止。"
                    job.finished_at = time()
                    return
                job.status = "failed"
                job.error = exc
                job.message = str(exc).strip() or exc.__class__.__name__
                job.finished_at = time()
            return
        finally:
            _job_context.manager = None
            _job_context.job_id = None
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job_id in self._cancel_requested:
                job.status = "canceled"
                job.message = "任务已终止；返回结果已丢弃。"
                job.finished_at = time()
                return
            job.status = "completed"
            job.progress = 100
            job.message = "处理完成"
            job.result = result
            job.finished_at = time()

    def set_metadata(self, job_id: str, metadata: dict[str, Any]) -> None:
        """Attach UI routing data so jobs survive Streamlit page reruns/sessions."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.metadata = dict(metadata)

    def set_estimate(self, job_id: str, seconds: float | None, note: str = "") -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.estimated_seconds = max(1.0, float(seconds)) if seconds else None
                job.estimate_note = str(note)

    def snapshots(self) -> list[JobSnapshot]:
        with self._lock:
            return [self._copy(job) for job in self._jobs.values()]

    def cancel(self, job_id: str) -> bool:
        """Request cancellation without blocking the Streamlit UI thread."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in {"completed", "failed", "canceled", "canceling"}:
                return False
            self._cancel_requested.add(job_id)
            job.cancel_requested_at = time()
            if job.status == "queued":
                job.status = "canceled"
                job.message = "任务已终止。"
                job.finished_at = time()
            else:
                job.status = "canceling"
                job.message = "已请求终止，正在等待当前模型调用释放资源……"
            return True

    @staticmethod
    def _copy(job: JobSnapshot) -> JobSnapshot:
        return JobSnapshot(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            message=job.message,
            result=job.result,
            error=job.error,
            metadata=dict(job.metadata) if job.metadata is not None else None,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            cancel_requested_at=job.cancel_requested_at,
            estimated_seconds=job.estimated_seconds,
            estimate_note=job.estimate_note,
        )

    def snapshot(self, job_id: str) -> JobSnapshot | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return self._copy(job)

    def discard(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
            self._cancel_requested.discard(job_id)


def check_current_job_cancelled() -> None:
    """Allow long streaming calls to cooperatively stop from inside their worker."""
    manager = getattr(_job_context, "manager", None)
    job_id = getattr(_job_context, "job_id", None)
    if manager is None or not job_id:
        return
    with manager._lock:
        if job_id in manager._cancel_requested or job_id not in manager._jobs:
            raise JobCancelledError("任务已由用户终止。")


@lru_cache(maxsize=1)
def get_job_manager() -> BackgroundJobManager:
    return BackgroundJobManager()
