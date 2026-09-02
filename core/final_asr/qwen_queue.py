import asyncio
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from loguru import logger

from core.final_asr.base import BaseFinalASREngine
from config.settings import FinalASRConfig
from core.metrics.prometheus_metrics import metrics


@dataclass
class FinalJobItem:
    audio_bytes: bytes
    context: Optional[str]
    hotwords: Optional[List[str]]
    enqueued_at: float
    future: asyncio.Future
    timeout_sec: float


class FinalQueue:
    """
    Asynchronous Micro-Batching Final Queue for Qwen3-ASR inference on Ascend NPU / vLLM.
    Groups sentence endpoints arriving within a short window (10~30ms) into micro-batches
    to maximize NPU utilization while keeping latency within P95 bounds.
    """

    def __init__(self, engine: BaseFinalASREngine, config: FinalASRConfig):
        self.engine = engine
        self.config = config
        self._queue: asyncio.Queue[FinalJobItem] = asyncio.Queue(maxsize=config.max_queue_size)
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._worker_task: Optional[asyncio.Task] = None
        self._is_running = False

        # Metrics counters
        self.total_jobs_submitted = 0
        self.total_jobs_completed = 0
        self.total_timeouts = 0
        self.total_fallbacks = 0
        self.last_batch_size = 0
        self.last_queue_wait_ms = 0.0
        self.last_inference_ms = 0.0
        # 回退率 EWMA (0~1): 二遍超时/溢出/失败占比, 准入控制探针。
        # 系数 0.98 -> 时间常数约 50 句。仅事件循环线程读写, 无需锁。
        self._fallback_ewma = 0.0

    @property
    def fallback_ewma(self) -> float:
        return self._fallback_ewma

    def _mark_fallback(self, fallback: bool):
        self._fallback_ewma = 0.98 * self._fallback_ewma + 0.02 * float(fallback)

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def start(self):
        if not self._is_running:
            self._is_running = True
            self._worker_task = asyncio.create_task(self._batch_dispatcher_loop())
            logger.info("[FinalQueue] Batch dispatcher worker started.")

    async def stop(self):
        self._is_running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        # Drain pending queue: fail all queued futures so callers don't hang
        drained = 0
        while not self._queue.empty():
            try:
                job = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not job.future.done():
                job.future.set_exception(asyncio.CancelledError("FinalQueue stopping"))
                drained += 1
        if drained:
            logger.info(f"[FinalQueue] Drained {drained} pending jobs on stop")
        metrics.set_gauge("asr_qwen_queue_size", 0)
        logger.info("[FinalQueue] Batch dispatcher worker stopped.")

    async def submit(
        self,
        audio_bytes: bytes,
        context: Optional[str] = None,
        hotwords: Optional[List[str]] = None,
        timeout_sec: Optional[float] = None,
    ) -> str:
        """
        Submit a speech segment for Final ASR.
        Returns the recognized final text or raises TimeoutError / Exception.
        """
        if not self._is_running:
            self.start()

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        effective_timeout = timeout_sec or self.config.hard_timeout_sec

        job = FinalJobItem(
            audio_bytes=audio_bytes,
            context=context,
            hotwords=hotwords,
            enqueued_at=time.time(),
            future=future,
            timeout_sec=effective_timeout,
        )

        self.total_jobs_submitted += 1
        try:
            self._queue.put_nowait(job)
            metrics.set_gauge("asr_qwen_queue_size", self._queue.qsize())
        except asyncio.QueueFull:
            # Overload / second-pass outage: reject instead of piling segment
            # audio in RAM. The caller falls back to the provisional text.
            self.total_fallbacks += 1
            self._mark_fallback(True)
            metrics.inc_counter("asr_qwen_fallback_total")
            logger.warning(
                f"[FinalQueue] Queue full ({self.config.max_queue_size} jobs), rejecting submit."
            )
            raise RuntimeError("Final queue overflow")

        try:
            result = await asyncio.wait_for(future, timeout=effective_timeout)
            self.total_jobs_completed += 1
            self._mark_fallback(False)
            return result
        except asyncio.TimeoutError:
            self.total_timeouts += 1
            self.total_fallbacks += 1
            self._mark_fallback(True)
            metrics.inc_counter("asr_qwen_timeout_total")
            metrics.inc_counter("asr_qwen_fallback_total")
            logger.warning(
                f"[FinalQueue] Job timed out after {effective_timeout}s, trigger fallback."
            )
            raise
        finally:
            metrics.set_gauge("asr_qwen_queue_size", self._queue.qsize())

    async def _batch_dispatcher_loop(self):
        """
        Micro-batching dispatcher loop.
        Collects up to max_batch_size items within batch_window_ms.
        """
        batch_window_sec = self.config.batch_window_ms / 1000.0

        while self._is_running:
            try:
                # Wait for first item
                first_job = await self._queue.get()
                batch: List[FinalJobItem] = [first_job]
                deadline = time.time() + batch_window_sec

                # Try collecting more items within the batch window
                while len(batch) < self.config.max_batch_size:
                    timeout = deadline - time.time()
                    if timeout <= 0:
                        break
                    try:
                        next_job = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                        batch.append(next_job)
                    except asyncio.TimeoutError:
                        break

                # Dispatch this batch
                asyncio.create_task(self._process_batch(batch))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[FinalQueue] Unexpected error in dispatcher: {e}")
                await asyncio.sleep(0.01)

    async def _process_batch(self, batch: List[FinalJobItem]):
        """
        Executes a batch of jobs with concurrency semaphore.
        """
        now = time.time()
        self.last_batch_size = len(batch)
        waits = [(now - j.enqueued_at) * 1000.0 for j in batch]
        self.last_queue_wait_ms = sum(waits) / len(waits) if waits else 0.0
        metrics.observe("asr_qwen_batch_size", float(self.last_batch_size))
        metrics.observe("asr_qwen_queue_wait_ms", self.last_queue_wait_ms)
        metrics.set_gauge("asr_qwen_queue_size", self._queue.qsize())

        # Filter out already cancelled or expired jobs
        active_batch: List[FinalJobItem] = []
        for job in batch:
            if job.future.done():
                continue
            if (now - job.enqueued_at) > job.timeout_sec:
                if not job.future.done():
                    job.future.set_exception(asyncio.TimeoutError("Queue wait timeout"))
            else:
                active_batch.append(job)

        if not active_batch:
            return

        # 按句占用并发额度并逐句执行: max_concurrency = 同时在途的二遍请求数。
        # (旧语义整批只占 1 个名额, 名为 8 实际可达 8x8=64 路, 且无法压低。)
        async def _run_one(job: FinalJobItem):
            async with self._semaphore:
                try:
                    return await self.engine.transcribe(
                        job.audio_bytes,
                        context=job.context,
                        hotwords=job.hotwords,
                    )
                except Exception as e:  # 单句失败不影响同批其他句
                    return e

        start_infer = time.time()
        results = await asyncio.gather(*[_run_one(j) for j in active_batch])
        self.last_inference_ms = (time.time() - start_infer) * 1000.0
        metrics.observe("asr_qwen_inference_ms", self.last_inference_ms)

        for job, res in zip(active_batch, results):
            if job.future.done():
                continue
            if isinstance(res, BaseException):
                job.future.set_exception(res)
                self._mark_fallback(True)
                metrics.inc_counter("asr_qwen_fallback_total")
            else:
                job.future.set_result(res)
