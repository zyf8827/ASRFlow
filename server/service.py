import asyncio
import logging
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from loguru import logger

from config.settings import AppConfig
from core.session import ClientSession
from core.session_manager import SessionManager
from core.vad import create_vad_engine
from core.streaming_asr import create_streaming_asr_engine
from core.final_asr import create_final_asr_engine, FinalQueue
from core.speaker import create_speaker_engine
from core.punc import create_punc_engine
from core.guard import ConsistencyGuard
from core.itn import ITNProcessor
from core.hotword import HotwordManager
from core.metrics.prometheus_metrics import metrics
from core.registry import NacosRegistry
from pipeline.session_pipeline import SessionPipeline
from server.admission import AdmissionController
from server.websocket_server import WebSocketGateway
from server.http_server import HTTPServer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _InterceptHandler(logging.Handler):
    """Route stdlib logging records (e.g. aiohttp access logs) into loguru
    so every sink (stdout + file) sees a single consistent format."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


class ASRRealtimeService:
    """
    Top-level industrial ASR Realtime Service orchestrator.
    Manages thread pools, model engines, final inference queue, and network servers.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig.load()
        self._setup_logging()

        logger.info("=" * 60)
        logger.info("Initializing ASRFlow Heterogeneous 2-Pass Service")
        logger.info("=" * 60)

        # Thread Pool for CPU-bound model tasks (VAD, Streaming ASR, Speaker)
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.pool.worker_threads,
            thread_name_prefix="asr_worker",
        )

        # Initialize Engines
        logger.info("Loading model engines...")
        self.vad_engine = create_vad_engine(self.config.vad)
        self.streaming_asr_engine = create_streaming_asr_engine(self.config.streaming_asr)
        self.final_asr_engine = create_final_asr_engine(self.config.final_asr)
        self.speaker_engine = create_speaker_engine(self.config.speaker)
        self.punc_engine = (
            create_punc_engine(self.config.punc) if self.config.punc.enable_realtime else None
        )

        # Initialize Components
        self.final_queue = FinalQueue(self.final_asr_engine, self.config.final_asr)
        self.guard = ConsistencyGuard(self.config.guard)
        self.itn = ITNProcessor(self.config.itn)
        self.hotword_mgr = HotwordManager()
        self.session_manager = SessionManager(self.config)

        # Admission control: EWMA probes maintained by the engines/queue,
        # evaluated by a hysteresis state machine (Mock engines expose no
        # probe -> that probe is simply absent, admission never trips)
        probes = {}
        wait_fn = getattr(self.streaming_asr_engine, "wait_ewma_ms", None)
        if callable(wait_fn):
            probes["streaming_wait_ms"] = (wait_fn, self.config.admission.streaming_wait_ms_limit)
        vad_fn = getattr(self.vad_engine, "lock_wait_ewma_ms", None)
        if callable(vad_fn):
            probes["vad_wait_ms"] = (vad_fn, self.config.admission.vad_wait_ms_limit)
        fb_fn = getattr(self.final_queue, "fallback_ewma", None)
        if callable(fb_fn):
            probes["final_fallback_rate"] = (fb_fn, self.config.admission.final_fallback_rate_limit)
        self.admission = AdmissionController(self.config.admission, probes)

        # Readiness probes: engines stuck in a self-healing reload loop must
        # flip /ready to 503 so the LB stops routing new traffic here
        readiness_probes = {}
        for name, engine in (
            ("streaming", self.streaming_asr_engine),
            ("vad", self.vad_engine),
        ):
            fn = getattr(engine, "is_available", None)
            if callable(fn):
                readiness_probes[name] = fn

        # Initialize Gateways
        self.ws_gateway = WebSocketGateway(
            session_manager=self.session_manager,
            pipeline_factory=self.create_pipeline,
            config=self.config,
            hotword_mgr=self.hotword_mgr,
            admission=self.admission,
        )
        self.http_server = HTTPServer(
            session_manager=self.session_manager,
            config=self.config,
            final_queue=self.final_queue,
            readiness_probes=readiness_probes,
            admission=self.admission,
        )

        # Optional Nacos service registration (client/discovery)
        self.registry: Optional[NacosRegistry] = None
        if self.config.registry.enable:
            self.registry = NacosRegistry(
                config=self.config.registry,
                service_port=self.config.server.port,
            )

        self._cleanup_task: Optional[asyncio.Task] = None
        self._lag_task: Optional[asyncio.Task] = None
        self._is_running = False

    def _setup_logging(self):
        logger.remove()
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        obs = self.config.observability
        logger.add(sys.stdout, level=obs.log_level, format=log_format, colorize=True)

        # Resolve the log file: explicit log_file wins; otherwise <log_dir>/
        # asrflow.log. Relative paths anchor to the project root so the file
        # lands in the repo's logs/ regardless of the launch directory.
        if obs.log_file:
            log_path = Path(obs.log_file)
            if not log_path.is_absolute():
                log_path = _PROJECT_ROOT / log_path
        elif obs.log_dir:
            log_dir = Path(obs.log_dir)
            if not log_dir.is_absolute():
                log_dir = _PROJECT_ROOT / log_dir
            log_path = log_dir / "asrflow.log"
        else:
            log_path = None

        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.add(
                str(log_path),
                level=obs.log_level,
                format=log_format,
                rotation=obs.log_rotation or None,
                retention=obs.log_retention or None,
                encoding="utf-8",
            )
            logger.info(
                f"[Logging] File sink: {log_path} "
                f"(rotation={obs.log_rotation or 'off'}, retention={obs.log_retention or 'forever'})"
            )

        # stdlib logging (aiohttp access log etc.) flows into the sinks above
        logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    def create_pipeline(self, session: ClientSession, output_queue: asyncio.Queue) -> SessionPipeline:
        return SessionPipeline(
            session=session,
            vad_engine=self.vad_engine,
            streaming_asr_engine=self.streaming_asr_engine,
            final_queue=self.final_queue,
            speaker_engine=self.speaker_engine,
            punc_engine=self.punc_engine,
            guard=self.guard,
            itn=self.itn,
            hotword_mgr=self.hotword_mgr,
            thread_pool=self.thread_pool,
            output_queue=output_queue,
            config=self.config,
        )

    async def _idle_session_cleaner(self):
        while self._is_running:
            try:
                await asyncio.sleep(60.0)
                max_idle = getattr(self.config.server, "idle_timeout_sec", 300.0)
                await self.session_manager.cleanup_idle_sessions(max_idle_sec=max_idle)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[ASRRealtimeService] Error in session cleanup loop: {e}")

    async def _loop_lag_monitor(self):
        """
        Event-loop lag watchdog: sleep overshoot is time the loop spent stuck
        in synchronous work. Catches "sync CPU/IO on the loop" regressions
        (they used to stall ALL sessions and trip WS ping timeouts).
        """
        interval = 0.2
        while self._is_running:
            t0 = time.monotonic()
            await asyncio.sleep(interval)
            lag_ms = (time.monotonic() - t0 - interval) * 1000.0
            if lag_ms > 0.0:
                metrics.observe("asr_event_loop_lag_ms", lag_ms)

    async def start(self):
        self._is_running = True
        # 1. Start Final Queue
        self.final_queue.start()

        # 2. Start HTTP server
        await self.http_server.start()

        # 3. Start WebSocket server
        await self.ws_gateway.start()

        # 4. Start background maintenance tasks
        self._cleanup_task = asyncio.create_task(self._idle_session_cleaner())
        self._lag_task = asyncio.create_task(self._loop_lag_monitor())
        self.admission.start()

        logger.info("ASR Realtime Service is fully ready and accepting connections.")

        # 5. Register into Nacos (after readiness; failures never block serving)
        if self.registry is not None:
            try:
                await self.registry.start()
            except Exception as e:
                logger.error(f"[ASRRealtimeService] Nacos registration failed: {e}")

    async def stop(self):
        logger.info("Stopping ASR Realtime Service...")
        self._is_running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._lag_task:
            self._lag_task.cancel()
        await self.admission.stop()

        if self.registry is not None:
            try:
                await self.registry.stop()
            except Exception as e:
                logger.warning(f"[ASRRealtimeService] Nacos deregister failed: {e}")

        await self.ws_gateway.stop()
        await self.http_server.stop()
        await self.final_queue.stop()

        # Release engine-owned resources (e.g. the Qwen engine's aiohttp
        # ClientSession) to avoid unclosed-session warnings and socket leaks
        close = getattr(self.final_asr_engine, "close", None)
        if close is not None:
            try:
                await close()
            except Exception as e:
                logger.warning(f"[ASRRealtimeService] Closing final engine failed: {e}")

        self.thread_pool.shutdown(wait=True)
        logger.info("ASR Realtime Service stopped cleanly.")

    async def run(self):
        await self.start()
        stop_event = asyncio.Event()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:
                pass

        try:
            await stop_event.wait()
        finally:
            await self.stop()
