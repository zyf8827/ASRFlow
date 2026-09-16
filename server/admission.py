import asyncio
import time
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger

from config.settings import AdmissionConfig
from core.metrics.prometheus_metrics import metrics

Probe = Callable[[], float]


class AdmissionController:
    """
    负载准入控制器: 引擎/队列侧维护 EWMA 探针, 本控制器周期评估,
    探针持续超限 -> 拒绝新建会话 (滞回状态机)。

    - 饱和判定: 任一探针 > limit 持续 saturate_sec;
    - 解除判定: 全部探针 < limit/2 持续 recover_sec (不对称阈值防抖动);
    - resume 重连已持有容量, 不经过本控制器;
    - enable=False 时永远放行 (评估循环只更新指标);
    - 热路径 allow_new_session() 只读快照, O(1) 无锁。

    探针 gauge 命名: asr_admission_<probe_name> (probe_name 即指标后缀,
    如 streaming_wait_ms / vad_wait_ms / final_fallback_rate)。
    """

    def __init__(
        self,
        config: AdmissionConfig,
        probes: Dict[str, Tuple[Probe, float]],
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self._probes = probes  # name -> (read_fn, limit)
        self._clock = clock
        self._saturated = False
        self._reasons: List[str] = []
        self._over_since: Optional[float] = None
        self._under_since: Optional[float] = None
        self._task: Optional[asyncio.Task] = None

    @property
    def saturated(self) -> bool:
        return self._saturated

    @property
    def reasons(self) -> List[str]:
        return list(self._reasons)

    def allow_new_session(self) -> Tuple[bool, List[str]]:
        """会话创建入口的准入门; True=放行。"""
        if not self.config.enable or not self._saturated:
            return True, []
        return False, self.reasons

    def evaluate(self) -> Tuple[bool, List[str]]:
        """单次评估并推进滞回状态机; 返回 (是否饱和, 原因)。"""
        over: List[str] = []
        all_below_half = True
        for name, (read, limit) in self._probes.items():
            try:
                value = float(read())
            except Exception:
                continue
            metrics.set_gauge(f"asr_admission_{name}", value)
            if value > limit:
                over.append(f"{name}={value:.1f}>{limit:g}")
            if value >= limit * 0.5:
                all_below_half = False

        now = self._clock()
        if over:
            self._under_since = None
            if self._over_since is None:
                self._over_since = now
            elif not self._saturated and now - self._over_since >= self.config.saturate_sec:
                self._saturated = True
                self._reasons = list(over)
                logger.warning(
                    f"[Admission] SATURATED, rejecting new sessions: {', '.join(over)}"
                )
        else:
            self._over_since = None
            if self._saturated:
                if all_below_half:
                    if self._under_since is None:
                        self._under_since = now
                    elif now - self._under_since >= self.config.recover_sec:
                        self._saturated = False
                        self._reasons = []
                        self._under_since = None
                        logger.info("[Admission] Recovered, accepting new sessions again")
                else:
                    self._under_since = None

        metrics.set_gauge("asr_admission_saturated", 1.0 if self._saturated else 0.0)
        return self._saturated, self.reasons

    async def run(self):
        """1s 周期评估 (由 service 作为后台任务启动)。"""
        while True:
            try:
                self.evaluate()
            except Exception as e:
                logger.error(f"[Admission] Evaluate failed: {e}")
            await asyncio.sleep(1.0)

    def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run())

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
