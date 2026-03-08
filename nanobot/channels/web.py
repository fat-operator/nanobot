"""Web UI channel — browser-based chat interface over HTTP + WebSocket."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import WebConfig

_WEB_UI_DIR = Path(__file__).parent / "web_ui"


class WebChannel(BaseChannel):
    """Serves a browser chat UI via HTTP and handles real-time messaging over WebSocket."""

    name = "web"

    def __init__(self, config: WebConfig, bus: MessageBus, *, session_manager=None, subagent_manager=None):
        super().__init__(config, bus)
        self.config: WebConfig = config
        self._session_manager = session_manager
        self._subagent_manager = subagent_manager
        # chat_id -> set of websocket connections
        self._clients: dict[str, set] = {}
        self._server = None
        self._index_html: bytes | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        import websockets

        self._running = True
        self._load_index_html()

        backoff = 1
        max_backoff = 60
        while self._running:
            try:
                self._server = await websockets.serve(
                    self._ws_handler,
                    self.config.host,
                    self.config.port,
                    process_request=self._process_http,
                )
                logger.info("Web UI listening on http://{}:{}", self.config.host, self.config.port)
                backoff = 1  # reset on success
                await asyncio.Future()  # run forever
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._running:
                    break
                logger.error("Web UI server crashed: {}. Retrying in {}s...", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    async def stop(self) -> None:
        self._running = False
        # Close all websocket connections
        for clients in self._clients.values():
            for ws in list(clients):
                try:
                    await asyncio.wait_for(ws.close(), timeout=2.0)
                except Exception:
                    pass
        self._clients.clear()
        if self._server:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Web UI server close timed out")

    # ------------------------------------------------------------------
    # HTTP handler (process_request hook)
    # ------------------------------------------------------------------

    async def _process_http(self, connection, request):
        """Handle plain HTTP requests; return None to continue with WS upgrade."""
        from websockets.datastructures import Headers
        from websockets.http11 import Response

        req_path = request.path
        hdrs = request.headers

        # Let WebSocket upgrades through
        if hdrs.get("Upgrade", "").lower() == "websocket":
            return None

        def _resp(status, reason, body, content_type="text/plain"):
            h = Headers([
                ("Content-Type", content_type),
                ("Content-Length", str(len(body))),
            ])
            return Response(status, reason, h, body)

        # GET /
        if req_path in ("/", ""):
            body = self._index_html or b"<h1>nanobot web UI</h1>"
            return _resp(200, "OK", body, "text/html; charset=utf-8")

        # GET /api/sessions
        if req_path == "/api/sessions":
            body = json.dumps(self._get_sessions_list()).encode()
            return _resp(200, "OK", body, "application/json")

        # GET /api/sessions/<key>/history
        if req_path.startswith("/api/sessions/") and req_path.endswith("/history"):
            session_key = unquote(req_path[len("/api/sessions/"):-len("/history")])
            body = json.dumps(self._get_session_history(session_key)).encode()
            return _resp(200, "OK", body, "application/json")

        # POST /api/upload — hint to use WebSocket upload instead
        if req_path == "/api/upload":
            body = json.dumps({"info": "Use WebSocket file upload"}).encode()
            return _resp(200, "OK", body, "application/json")

        return _resp(404, "Not Found", b"Not Found")

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def _ws_handler(self, websocket):
        """Handle a WebSocket connection."""
        chat_id = str(uuid.uuid4())
        if chat_id not in self._clients:
            self._clients[chat_id] = set()
        self._clients[chat_id].add(websocket)
        logger.info("Web UI client connected: {}", chat_id)

        try:
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self._ws_send(websocket, {"type": "error", "content": "Invalid JSON"})
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "chat":
                    content = msg.get("content", "").strip()
                    session_key = msg.get("session_key") or f"web:{chat_id}"
                    media = msg.get("media") or []

                    # Move client to correct chat_id bucket (same pattern as session_switch)
                    new_id = session_key.split(":", 1)[-1] if ":" in session_key else session_key
                    if new_id != chat_id:
                        if chat_id in self._clients:
                            self._clients[chat_id].discard(websocket)
                            if not self._clients[chat_id]:
                                del self._clients[chat_id]
                        chat_id = new_id
                        if chat_id not in self._clients:
                            self._clients[chat_id] = set()
                        self._clients[chat_id].add(websocket)

                    if content:
                        await self._handle_message(
                            sender_id="web_user",
                            chat_id=chat_id,
                            content=content,
                            media=media if media else None,
                            session_key=session_key,
                        )

                elif msg_type == "sessions_list":
                    data = self._get_sessions_list()
                    await self._ws_send(websocket, {"type": "sessions_list", "data": data})

                elif msg_type == "session_switch":
                    new_key = msg.get("session_key", "")
                    # Move client to new chat_id bucket
                    if chat_id in self._clients:
                        self._clients[chat_id].discard(websocket)
                        if not self._clients[chat_id]:
                            del self._clients[chat_id]
                    new_id = new_key.split(":", 1)[-1] if ":" in new_key else new_key
                    chat_id = new_id
                    if chat_id not in self._clients:
                        self._clients[chat_id] = set()
                    self._clients[chat_id].add(websocket)

                    history = self._get_session_history(new_key)
                    await self._ws_send(websocket, {
                        "type": "history",
                        "messages": history,
                        "session_key": new_key,
                    })

                elif msg_type == "new_session":
                    # Move client to a fresh session
                    if chat_id in self._clients:
                        self._clients[chat_id].discard(websocket)
                        if not self._clients[chat_id]:
                            del self._clients[chat_id]
                    chat_id = str(uuid.uuid4())
                    if chat_id not in self._clients:
                        self._clients[chat_id] = set()
                    self._clients[chat_id].add(websocket)
                    await self._ws_send(websocket, {
                        "type": "new_session",
                        "session_key": f"web:{chat_id}",
                    })

                elif msg_type == "session_delete":
                    del_key = msg.get("session_key", "")
                    deleted = False
                    if del_key and self._session_manager:
                        # Cancel any running subagents for this session
                        if self._subagent_manager:
                            try:
                                await self._subagent_manager.cancel_by_session(del_key)
                            except Exception as e:
                                logger.warning("Failed to cancel subagents for session {}: {}", del_key, e)
                        deleted = self._session_manager.delete(del_key)
                    await self._ws_send(websocket, {
                        "type": "session_deleted",
                        "session_key": del_key,
                        "deleted": deleted,
                    })
                    # Push updated sessions list
                    data = self._get_sessions_list()
                    await self._ws_send(websocket, {"type": "sessions_list", "data": data})

                elif msg_type == "upload":
                    # Handle base64 file upload via WebSocket
                    import base64
                    filename = msg.get("filename", "upload")
                    file_data = msg.get("data", "")
                    try:
                        raw_bytes = base64.b64decode(file_data)
                        upload_dir = self._get_upload_dir()
                        safe_name = filename.replace("/", "_").replace("\\", "_")
                        dest = (upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}").resolve()
                        if not str(dest).startswith(str(upload_dir.resolve())):
                            raise ValueError("Invalid filename")
                        dest.write_bytes(raw_bytes)
                        await self._ws_send(websocket, {
                            "type": "upload_ok",
                            "path": str(dest),
                            "filename": safe_name,
                        })
                    except Exception as e:
                        await self._ws_send(websocket, {
                            "type": "error",
                            "content": f"Upload failed: {e}",
                        })

                elif msg_type == "subagents":
                    sk = msg.get("session_key", "")
                    tasks = self._get_subagent_tasks(sk)
                    await self._ws_send(websocket, {
                        "type": "subagents",
                        "session_key": sk,
                        "tasks": tasks,
                    })

                elif msg_type == "subagent_detail":
                    detail = self._get_subagent_detail(msg.get("task_id", ""))
                    await self._ws_send(websocket, {
                        "type": "subagent_detail",
                        **detail,
                    })

                else:
                    await self._ws_send(websocket, {"type": "error", "content": f"Unknown type: {msg_type}"})

        except Exception as e:
            if "closed" not in str(e).lower() and "going away" not in str(e).lower():
                logger.warning("Web UI client error: {}", e)
        finally:
            # Clean up
            for cid, clients in list(self._clients.items()):
                clients.discard(websocket)
                if not clients:
                    del self._clients[cid]
            logger.info("Web UI client disconnected")

    # ------------------------------------------------------------------
    # Outbound: send() override
    # ------------------------------------------------------------------

    async def send(self, msg: OutboundMessage) -> None:
        """Route an outbound message to connected WebSocket clients."""
        chat_id = msg.chat_id
        clients = self._clients.get(chat_id, set())
        if not clients:
            return

        if msg.metadata.get("_progress"):
            if msg.metadata.get("_tool_hint"):
                ws_type = "tool_hint"
            else:
                ws_type = "progress"
        else:
            ws_type = "chat"

        payload = {
            "type": ws_type,
            "content": msg.content,
            "session_key": f"web:{chat_id}",
        }

        dead = set()
        for ws in clients:
            try:
                await ws.send(json.dumps(payload))
            except Exception:
                dead.add(ws)
        if dead and chat_id in self._clients:
            self._clients[chat_id] -= dead

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_index_html(self) -> None:
        html_path = _WEB_UI_DIR / "index.html"
        if html_path.exists():
            self._index_html = html_path.read_bytes()
        else:
            logger.warning("Web UI index.html not found at {}", html_path)

    def _get_sessions_list(self) -> list[dict[str, Any]]:
        if not self._session_manager:
            return []
        all_sessions = self._session_manager.list_sessions()
        internal_prefixes = ("cron:", "heartbeat", "system:")
        visible_channels = self.config.visible_channels
        sessions = []
        for s in all_sessions:
            key = s.get("key", "")
            if key.startswith(internal_prefixes):
                continue
            if visible_channels and not any(key.startswith(p + ":") for p in visible_channels):
                continue
            sessions.append({"key": key, "created_at": s.get("created_at"), "updated_at": s.get("updated_at")})
        return sessions

    def _get_session_history(self, session_key: str) -> list[dict[str, Any]]:
        if not self._session_manager:
            return []
        session = self._session_manager.get(session_key)
        if session is None:
            return []
        messages = []
        for m in session.messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        return messages

    def _get_upload_dir(self) -> Path:
        if self._session_manager:
            upload_dir = self._session_manager.workspace / "uploads"
        else:
            upload_dir = Path.home() / ".nanobot" / "workspace" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        return upload_dir

    def _get_subagent_detail(self, task_id: str) -> dict[str, Any]:
        if not self._subagent_manager:
            return {"task_id": task_id, "label": "", "status": "", "steps": []}
        meta = self._subagent_manager.get_task_detail(task_id)
        return {
            "task_id": task_id,
            "label": meta.get("label", ""),
            "status": meta.get("status", ""),
            "steps": meta.get("steps", []),
        }

    def _get_subagent_tasks(self, session_key: str) -> list[dict[str, Any]]:
        if not self._subagent_manager:
            return []
        return self._subagent_manager.get_tasks(session_key)

    async def notify_subagent_update(
        self, session_key: str, task_id: str, label: str, status: str,
    ) -> None:
        """Push a subagent state change to all connected clients for this session."""
        chat_id = session_key.split(":", 1)[-1] if ":" in session_key else session_key
        clients = self._clients.get(chat_id, set())
        payload = json.dumps({
            "type": "subagent_update",
            "session_key": session_key,
            "task_id": task_id,
            "label": label,
            "status": status,
        })
        for ws in list(clients):
            try:
                await ws.send(payload)
            except Exception:
                pass

    @staticmethod
    async def _ws_send(ws, data: dict) -> None:
        try:
            await ws.send(json.dumps(data))
        except Exception:
            pass
