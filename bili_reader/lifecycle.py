from __future__ import annotations

import base64
import hashlib
import os
from socketserver import BaseRequestHandler, ThreadingMixIn, TCPServer
from threading import Lock, Thread
from time import monotonic, sleep
from urllib.parse import unquote, urlparse

import streamlit.components.v1 as components
from streamlit.runtime import Runtime


_state_lock = Lock()
_start_lock = Lock()
_clients: set[str] = set()
_saw_client = False
_socket_server: "_ThreadingTCPServer | None" = None
_server_thread: Thread | None = None
_monitor_thread: Thread | None = None


def _register_client(client_id: str) -> None:
    global _saw_client
    with _state_lock:
        _clients.add(client_id)
        _saw_client = True


def _remove_client(client_id: str) -> None:
    with _state_lock:
        _clients.discard(client_id)


class _ThreadingTCPServer(ThreadingMixIn, TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _WebSocketHandler(BaseRequestHandler):
    def handle(self) -> None:
        client_id = ""
        try:
            request = bytearray()
            while b"\r\n\r\n" not in request and len(request) < 16_384:
                block = self.request.recv(2_048)
                if not block:
                    return
                request.extend(block)
            lines = bytes(request).decode("latin-1", errors="replace").split("\r\n")
            request_line = lines[0].split()
            if len(request_line) < 2 or request_line[0] != "GET":
                return
            client_id = unquote(urlparse(request_line[1]).path.strip("/"))
            headers = {
                key.strip().lower(): value.strip()
                for line in lines[1:]
                if ":" in line
                for key, value in [line.split(":", 1)]
            }
            websocket_key = headers.get("sec-websocket-key", "")
            if not client_id or not websocket_key:
                return
            accept = base64.b64encode(
                hashlib.sha1(
                    (websocket_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
                ).digest()
            ).decode("ascii")
            response = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
            )
            self.request.sendall(response.encode("ascii"))
            _register_client(client_id)
            while self.request.recv(1_024):
                # The page sends no application data. Any frame is a close/control
                # frame, so returning closes the socket and unregisters the tab.
                break
        except (ConnectionError, OSError, UnicodeError):
            return
        finally:
            if client_id:
                _remove_client(client_id)


def _monitor_clients(
    disconnect_grace_seconds: float = 6.0,
    poll_interval: float = 0.5,
) -> None:
    empty_since: float | None = None
    while True:
        now = monotonic()
        with _state_lock:
            saw_client = _saw_client
            has_clients = bool(_clients)
        if has_clients:
            empty_since = None
        elif saw_client:
            if empty_since is None:
                empty_since = now
            elif now - empty_since >= disconnect_grace_seconds:
                if Runtime.exists():
                    Runtime.instance().stop()
                # Runtime.stop() does not stop Streamlit's outer Starlette server.
                # End this app process as well so queued workers and media children
                # cannot remain after the final browser tab has closed.
                sleep(0.25)
                os._exit(0)
                return
        sleep(poll_interval)


def start_browser_close_monitor() -> int:
    """Start one process-wide local WebSocket monitor and return its port."""
    global _socket_server, _server_thread, _monitor_thread
    with _start_lock:
        if _socket_server is None:
            _socket_server = _ThreadingTCPServer(("127.0.0.1", 0), _WebSocketHandler)
            _server_thread = Thread(
                target=_socket_server.serve_forever,
                name="bilibili-browser-socket-server",
                daemon=True,
            )
            _server_thread.start()
        if _monitor_thread is None or not _monitor_thread.is_alive():
            _monitor_thread = Thread(
                target=_monitor_clients,
                name="bilibili-browser-close-monitor",
                daemon=True,
            )
            _monitor_thread.start()
        return int(_socket_server.server_address[1])


def render_browser_heartbeat(port: int) -> None:
    """Hold one socket per browser tab; closing the tab closes the socket."""
    components.html(
        f"""
        <script>
        (() => {{
          const id = (globalThis.crypto && crypto.randomUUID)
            ? crypto.randomUUID()
            : `${{Date.now()}}-${{Math.random().toString(16).slice(2)}}`;
          globalThis.bilibiliReaderSocket = new WebSocket(`ws://127.0.0.1:{port}/${{id}}`);
        }})();
        </script>
        """,
        height=0,
        width=0,
    )
