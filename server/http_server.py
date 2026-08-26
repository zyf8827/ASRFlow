import time
from pathlib import Path
from typing import Callable, Dict, Optional
from aiohttp import web
from loguru import logger

from core.session_manager import SessionManager
from core.metrics.prometheus_metrics import metrics
from config.settings import AppConfig

_STATIC_DIR = Path(__file__).resolve().parent / "static"


class HTTPServer:
    """
    HTTP server exposing Health Check, Readiness, Prometheus Metrics, and Management API.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        config: AppConfig,
        final_queue=None,
        readiness_probes: Optional[Dict[str, Callable[[], bool]]] = None,
        admission=None,
    ):
        self.session_manager = session_manager
        self.config = config
        self.final_queue = final_queue
        # 引擎健康探针 (SelfHealingMixin.is_available); 引擎重载循环中
        # /ready 应返回 503 让 LB 摘流, 而不是继续接新流量
        self.readiness_probes = readiness_probes or {}
        self.admission = admission
        self.start_time = time.time()
        self._runner = None
        self._site = None

    async def handle_healthz(self, request: web.Request) -> web.Response:
        uptime = time.time() - self.start_time
        return web.json_response(
            {
                "status": "healthy",
                "uptime_seconds": round(uptime, 2),
                "active_sessions": self.session_manager.active_session_count,
            }
        )

    async def handle_ready(self, request: web.Request) -> web.Response:
        # Check final queue liveness if available
        queue_ok = True
        reason = None
        if self.final_queue is not None:
            try:
                is_running = getattr(self.final_queue, "_is_running", True)
                worker = getattr(self.final_queue, "_worker_task", None)
                if not is_running or (worker is not None and worker.done()):
                    queue_ok = False
                    reason = "final queue not running"
            except Exception as e:
                queue_ok = False
                reason = str(e)

        # Engine health probes (e.g. SelfHealingMixin.is_available): an engine
        # stuck in its reload loop must pull this instance out of the LB pool
        engine_status: Dict[str, bool] = {}
        for name, probe in self.readiness_probes.items():
            try:
                engine_status[name] = bool(probe())
            except Exception as e:
                engine_status[name] = False
                if reason is None:
                    reason = f"{name} engine probe failed: {e}"
        unavailable = [n for n, ok in engine_status.items() if not ok]

        if not queue_ok or unavailable:
            if unavailable:
                reason = reason or f"engines unavailable: {', '.join(unavailable)}"
            return web.json_response(
                {
                    "ready": False,
                    "reason": reason,
                    "service": "asrflow",
                    "ws_port": self.config.server.port,
                    "engines": engine_status,
                },
                status=503,
            )

        # Optional deep check: ?deep=true probes vLLM reachability (non-blocking for readiness)
        checks: dict = {}
        if request.query.get("deep") == "true":
            try:
                import aiohttp, asyncio

                timeout = aiohttp.ClientTimeout(total=2.0)
                async with aiohttp.ClientSession(timeout=timeout) as sess:
                    async with sess.head(self.config.final_asr.vllm_url, allow_redirects=True) as resp:
                        checks["vllm_reachable"] = resp.status < 500
                        checks["vllm_status"] = resp.status
            except Exception as e:
                checks["vllm_reachable"] = False
                checks["vllm_error"] = str(e)[:200]

        body = {
            "ready": True,
            "service": "asrflow",
            "ws_port": self.config.server.port,
            "streaming_asr": self.config.streaming_asr.backend,
            "vad": self.config.vad.backend,
            "final_asr": self.config.final_asr.engine_type,
        }
        if engine_status:
            body["engines"] = engine_status
        # Admission saturation is informational only: a saturated instance is
        # still serving its existing sessions, rejection happens per session
        if self.admission is not None:
            try:
                body["admission"] = {
                    "saturated": bool(getattr(self.admission, "saturated", False)),
                    "reasons": list(getattr(self.admission, "reasons", []) or []),
                }
            except Exception:
                pass
        if checks:
            body["checks"] = checks
        return web.json_response(body)

    async def handle_metrics(self, request: web.Request) -> web.Response:
        # Update dynamic gauge values
        metrics.set_gauge("asr_online_sessions", self.session_manager.active_session_count)
        if self.final_queue is not None:
            try:
                metrics.set_gauge("asr_qwen_queue_size", self.final_queue.queue_size)
            except Exception:
                pass
        text_data = metrics.export_text()
        return web.Response(text=text_data, content_type="text/plain", charset="utf-8")

    async def handle_sessions(self, request: web.Request) -> web.Response:
        sessions = await self.session_manager.list_sessions()
        return web.json_response({"total": len(sessions), "sessions": sessions})

    async def handle_dashboard(self, request: web.Request) -> web.StreamResponse:
        """Serve the vanilla-JS metrics dashboard (server/static/dashboard.html)."""
        html_path = _STATIC_DIR / "dashboard.html"
        if not html_path.is_file():
            return web.Response(status=500, text="dashboard.html is missing")
        return web.FileResponse(html_path)

    def create_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self.handle_dashboard)
        app.router.add_get("/dashboard", self.handle_dashboard)
        app.router.add_get("/healthz", self.handle_healthz)
        app.router.add_get("/ready", self.handle_ready)
        app.router.add_get("/metrics", self.handle_metrics)
        app.router.add_get("/api/v1/sessions", self.handle_sessions)
        return app

    async def start(self):
        host = self.config.server.host
        port = self.config.server.http_port
        app = self.create_app()
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, host, port)
        await self._site.start()
        logger.info(f"[HTTPServer] HTTP API & Metrics server running at http://{host}:{port} (dashboard: /dashboard)")

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()
            logger.info("[HTTPServer] HTTP server stopped.")
