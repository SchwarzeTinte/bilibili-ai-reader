from __future__ import annotations

import unittest
from unittest.mock import patch

from bili_reader import lifecycle


class _FakeRuntimeInstance:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _FakeRuntime:
    instance_value = _FakeRuntimeInstance()

    @classmethod
    def exists(cls) -> bool:
        return True

    @classmethod
    def instance(cls) -> _FakeRuntimeInstance:
        return cls.instance_value


class BrowserCloseMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        lifecycle._clients.clear()
        lifecycle._saw_client = False
        _FakeRuntime.instance_value = _FakeRuntimeInstance()

    def test_last_browser_close_stops_runtime_after_grace_period(self) -> None:
        lifecycle._register_client("tab-1")
        lifecycle._remove_client("tab-1")
        with (
            patch.object(lifecycle, "Runtime", _FakeRuntime),
            patch.object(lifecycle, "sleep"),
            patch.object(lifecycle, "monotonic", side_effect=[10.0, 17.0]),
            patch.object(lifecycle.os, "_exit") as exit_mock,
        ):
            lifecycle._monitor_clients(disconnect_grace_seconds=6.0)
        self.assertTrue(_FakeRuntime.instance_value.stopped)
        exit_mock.assert_called_once_with(0)

    def test_another_open_tab_prevents_shutdown(self) -> None:
        lifecycle._register_client("tab-1")
        lifecycle._register_client("tab-2")
        lifecycle._remove_client("tab-1")
        with lifecycle._state_lock:
            self.assertEqual(set(lifecycle._clients), {"tab-2"})

    def test_refresh_can_register_a_replacement_before_grace_expires(self) -> None:
        lifecycle._register_client("old-frame")
        lifecycle._remove_client("old-frame")
        lifecycle._register_client("new-frame")
        with lifecycle._state_lock:
            self.assertTrue(lifecycle._clients)


if __name__ == "__main__":
    unittest.main()
