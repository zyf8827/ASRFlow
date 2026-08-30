import json
import time
import asyncio
from typing import Optional
import websockets
from websockets.exceptions import ConnectionClosed
from loguru import logger

from core.session import ClientSession
from core.session_manager import SessionManager
from core.hotword.hotword_manager import HotwordManager
from core.metrics.prometheus_metrics import metrics
from pipeline.session_pipeline import SessionPipeline
from server.admission import AdmissionController
from config.settings import AppConfig

# Malformed-frame log rate limit: warn at most once per 5s (metric always counts)
_MALFORMED_WARN_INTERVAL_SEC = 5.0
_last_malformed_warn_ts = 0.0


class WebSocketGateway:
    """
    Industrial-grade WebSocket server for real-time audio streaming and recognition results.
    Separates JSON control protocol from raw PCM16 binary audio frames.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        pipeline_factory,
        config: AppConfig,
        hotword_mgr: Optional[HotwordManager] = None,
        admission: Optional[AdmissionController] = None,
    ):
        self.session_manager = session_manager
        self.pipeline_factory = pipeline_factory
        self.config = config
        self.hotword_mgr = hotword_mgr
        self.admission = admission
        self._server = None
        self._is_running = False

    # ------------------------------------------------------------------
    # Input validation & admission
    # ------------------------------------------------------------------
    def _validate_audio_frame(self, data: bytes) -> Optional[bytes]:
        """
        Binary frame gate: truncate odd-length (half sample) and oversized
        frames so one malformed message can never crash the engines or occupy
        the session loop for minutes. Returns None for empty frames.
        """
        global _last_malformed_warn_ts

        if len(data) & 1:
            data = data[:-1]
            metrics.inc_counter("asr_malformed_frames_total")
            now = time.monotonic()
            if now - _last_malformed_warn_ts > _MALFORMED_WARN_INTERVAL_SEC:
                _last_malformed_warn_ts = now
                logger.warning(
                    "[WebSocketGateway] Odd-length audio frame truncated "
                    f"(len={len(data) + 1}B)"
                )
        max_bytes = self.config.audio.bytes_per_ms * self.config.server.max_audio_frame_ms
        if len(data) > max_bytes:
            data = data[:max_bytes]
            metrics.inc_counter("asr_malformed_frames_total")
            now = time.monotonic()
            if now - _last_malformed_warn_ts > _MALFORMED_WARN_INTERVAL_SEC:
                _last_malformed_warn_ts = now
                logger.warning(
                    "[WebSocketGateway] Oversized audio frame truncated to "
                    f"{self.config.server.max_audio_frame_ms}ms"
                )
        return data or None

    async def _process_audio_safe(self, pipeline: SessionPipeline, frame: bytes):
        """
        Per-message exception isolation: one bad frame (or an engine hiccup on
        it) is skipped with a counter, never escalates to connection teardown.
        """
        try:
            await pipeline.process_audio_chunk(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            metrics.inc_counter("asr_chunk_processing_errors_total")
            logger.exception("[WebSocketGateway] Audio chunk processing failed; frame skipped")

    async def _admit_new_session(self, websocket, output_queue: asyncio.Queue) -> bool:
        """
        Admission gate for new sessions. On rejection: emit an error event and
        close with 1013 (Try Again Later). Returns False (caller must stop
        processing this connection).
        """
        if self.admission is None:
            return True
        ok, reasons = self.admission.allow_new_session()
        if ok:
            return True
        metrics.inc_counter("asr_sessions_rejected_overload_total")
        detail = ", ".join(reasons) if reasons else "load"
        await output_queue.put(
            {
                "type": "error",
                "event": "error",
                "code": 4290,
                "message": f"server overloaded ({detail}), retry later",
            }
        )
        # Give the send loop a chance to flush the error before the close frame
        await asyncio.sleep(0.05)
        await websocket.close(code=1013, reason="overloaded")
        return False

    async def handle_connection(self, websocket):
        """
        Handles an individual client WebSocket connection.
        """
        metrics.inc_gauge("asr_online_sessions")
        metrics.inc_counter("asr_total_sessions")
        logger.info(f"[WebSocketGateway] New connection from {websocket.remote_address}")

        session: Optional[ClientSession] = None
        pipeline: Optional[SessionPipeline] = None
        output_queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        send_task: Optional[asyncio.Task] = None

        try:
            # Start sender coroutine
            send_task = asyncio.create_task(self._send_loop(websocket, output_queue))

            async for message in websocket:
                if isinstance(message, bytes):
                    # Binary Audio Frame
                    if session:
                        session.touch()
                    frame = self._validate_audio_frame(message)
                    if frame is None:
                        continue
                    if not (pipeline and session and session.is_active):
                        # Auto-create default session if client directly streams audio before START
                        if not session:
                            if not await self._admit_new_session(websocket, output_queue):
                                break
                            logger.warning("[WebSocketGateway] Binary audio before START: auto-creating session")
                            session = await self.session_manager.create_session()
                            pipeline = self.pipeline_factory(session, output_queue)
                            await output_queue.put(
                                {
                                    "type": "session_ready",
                                    "session_id": session.session_id,
                                    "auto_created": True,
                                }
                            )
                        if session:
                            session.touch()
                    await self._process_audio_safe(pipeline, frame)

                elif isinstance(message, str):
                    raw_str = message.strip()
                    if not raw_str:
                        continue
                    # Touch on any WS activity to avoid idle eviction of live connections
                    if session:
                        session.touch()

                    # 1. Plain string commands (FunASR realtime_ws compatibility)
                    upper_str = raw_str.upper()
                    if upper_str == "PING":
                        await output_queue.put({"type": "pong", "event": "pong"})
                        continue
                    elif upper_str == "START":
                        if not session:
                            if not await self._admit_new_session(websocket, output_queue):
                                break
                            session = await self.session_manager.create_session()
                            pipeline = self.pipeline_factory(session, output_queue)
                        else:
                            session.is_active = True
                        await output_queue.put(
                            {
                                "type": "session_ready",
                                "event": "started",
                                "session_id": session.session_id,
                            }
                        )
                        continue
                    elif upper_str == "COMMIT":
                        if pipeline:
                            await pipeline.process_commit()
                        else:
                            await output_queue.put(
                                {"type": "error", "event": "error", "code": 4004, "message": "No active session to commit"}
                            )
                        continue
                    elif upper_str == "STOP":
                        if pipeline:
                            await pipeline.process_stop()
                        await output_queue.put({"event": "stopped"})
                        continue
                    elif upper_str.startswith("HOTWORDS:"):
                        hw_str = raw_str[9:].strip()
                        hotwords = [w.strip() for w in hw_str.split(",") if w.strip()]
                        if session:
                            session.hotwords = hotwords
                        if pipeline:
                            # Refresh the Pass-1 Paraformer biasing as well
                            pipeline.update_hotwords(hotwords)
                        await output_queue.put({"event": "hotwords_set", "hotwords": hotwords})
                        continue
                    elif upper_str.startswith("POSTPROCESS_HOTWORDS:"):
                        payload = raw_str.split(":", 1)[1]
                        if self.hotword_mgr is not None:
                            parsed_dict = self.hotword_mgr.parse_postprocess_hotwords_payload(payload)
                        else:
                            parsed_dict = {}
                        if session:
                            session.postprocess_hotwords = parsed_dict
                        await output_queue.put({"event": "postprocess_hotwords_set", "hotwords": parsed_dict})
                        continue
                    elif upper_str.startswith("LANGUAGE:"):
                        lang = raw_str[9:].strip()
                        if session:
                            session.language = lang
                        await output_queue.put({"event": "language_set", "language": lang})
                        continue

                    # 2. JSON structured commands (tech_plan.md standard)
                    try:
                        cmd = json.loads(raw_str)
                    except Exception as e:
                        logger.warning(f"[WebSocketGateway] Invalid JSON/command received: {raw_str} ({e})")
                        await output_queue.put(
                            {"type": "error", "code": 4000, "message": "Invalid command format"}
                        )
                        continue

                    msg_type = cmd.get("type", "").lower()

                    if msg_type == "start":
                        session_id = cmd.get("session_id")
                        language = cmd.get("language", "zh")
                        enable_spk = cmd.get("enable_spk", True)
                        expected_speakers = cmd.get("expected_speakers", 2)
                        hotwords = cmd.get("hotwords", [])

                        # Session resume: reattach a detached session kept
                        # alive after an unexpected disconnect
                        resumed_session = None
                        if cmd.get("resume") and session_id:
                            resumed_session = await self.session_manager.reattach_session(session_id)

                        if resumed_session is not None:
                            session = resumed_session
                            if "hotwords" in cmd:
                                session.hotwords = hotwords
                            pipeline = self.pipeline_factory(session, output_queue)

                            committed = session.commit_transcript()
                            last_end_ms = max((s["end_ms"] for s in committed), default=0)
                            logger.info(
                                f"[WebSocketGateway] Session resumed: {session.session_id} "
                                f"({len(committed)} committed sentences, last_end_ms={last_end_ms})"
                            )
                            await output_queue.put(
                                {
                                    "type": "session_resumed",
                                    "event": "resumed",
                                    "session_id": session.session_id,
                                    "last_end_ms": last_end_ms,
                                    "total_sentences": len(committed),
                                    "sentences": committed,
                                    "is_final": False,
                                }
                            )
                        else:
                            # Brand-new session goes through admission; the
                            # resume branch above deliberately bypasses it (a
                            # resuming session already owns its capacity)
                            if not await self._admit_new_session(websocket, output_queue):
                                break
                            session = await self.session_manager.create_session(
                                session_id=session_id,
                                language=language,
                                enable_spk=enable_spk,
                                expected_speakers=expected_speakers,
                                hotwords=hotwords,
                            )
                            pipeline = self.pipeline_factory(session, output_queue)

                            logger.info(
                                f"[WebSocketGateway] Session initialized: {session.session_id} "
                                f"(spk={enable_spk}, exp_spk={expected_speakers}, hotwords={len(hotwords)})"
                            )
                            await output_queue.put(
                                {
                                    "type": "session_ready",
                                    "event": "started",
                                    "session_id": session.session_id,
                                }
                            )

                    elif msg_type == "commit":
                        if pipeline:
                            await pipeline.process_commit()
                        else:
                            await output_queue.put(
                                {"type": "error", "code": 4004, "message": "No active session to commit"}
                            )

                    elif msg_type == "stop":
                        if pipeline:
                            await pipeline.process_stop()
                        else:
                            await output_queue.put(
                                {"type": "error", "code": 4004, "message": "No active session to stop"}
                            )

                    elif msg_type == "ping":
                        await output_queue.put({"type": "pong", "event": "pong"})

                    else:
                        logger.warning(f"[WebSocketGateway] Unknown command type: {msg_type}")
                        await output_queue.put(
                            {"type": "error", "code": 4002, "message": f"Unknown command '{msg_type}'"}
                        )

        except ConnectionClosed as e:
            logger.info(f"[WebSocketGateway] Connection closed by client: code={e.code}, reason={e.reason}")
        except Exception as e:
            metrics.inc_counter("asr_websocket_errors")
            logger.error(f"[WebSocketGateway] Unexpected error in connection loop: {e}")
        finally:
            metrics.dec_gauge("asr_online_sessions")
            if session:
                # Unexpected disconnect: keep the session resumable for the
                # configured grace period instead of tearing it down
                detached = False
                resume_ttl = self.config.server.resume_ttl_sec
                if pipeline and not session.is_stopping and resume_ttl > 0:
                    detached = await self.session_manager.detach_session(
                        session.session_id, pipeline
                    )
                    if detached:
                        logger.info(
                            f"[WebSocketGateway] Connection lost; session {session.session_id} "
                            f"kept resumable for {resume_ttl}s."
                        )

                if not detached:
                    if pipeline and not session.is_stopping:
                        try:
                            await asyncio.wait_for(
                                pipeline.process_stop(),
                                timeout=self.config.final_asr.hard_timeout_sec + 2.0,
                            )
                        except Exception:
                            pass
                    await self.session_manager.remove_session(session.session_id)

            if send_task and not send_task.done():
                send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass

            logger.info("[WebSocketGateway] Connection cleanup finished.")

    async def _send_loop(self, websocket, queue: asyncio.Queue):
        """
        Pushes results from output_queue to the client websocket connection.
        """
        while True:
            try:
                msg = await queue.get()
                if msg is None:
                    break
                if isinstance(msg, dict):
                    msg_text = json.dumps(msg, ensure_ascii=False)
                    await websocket.send(msg_text)
                elif isinstance(msg, str):
                    await websocket.send(msg)
                elif isinstance(msg, bytes):
                    await websocket.send(msg)
            except ConnectionClosed:
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[WebSocketGateway] Error sending message: {e}")
                break

    async def start(self):
        host = self.config.server.host
        port = self.config.server.port
        logger.info(f"[WebSocketGateway] Starting WebSocket server on ws://{host}:{port}...")

        self._server = await websockets.serve(
            self.handle_connection,
            host=host,
            port=port,
            max_size=self.config.server.max_message_size,
            ping_interval=self.config.server.ping_interval,
            ping_timeout=self.config.server.ping_timeout,
        )
        self._is_running = True
        logger.info(f"[WebSocketGateway] Server listening on ws://{host}:{port}")

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._is_running = False
            logger.info("[WebSocketGateway] Server stopped.")
