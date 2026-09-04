#!/usr/bin/env python3
"""
网关级 WebSocket 并发压测 —— 支持多实例负载均衡 (客户端轮询分发)。

目的: 在真实网关链路 (WS 协议 -> VAD -> 首遍 Paraformer -> ...) 上测并发承载,
并验证"多进程部署 + 负载均衡"的横向扩展效果:
  - 单实例: 会话并发从 1 逐步往上加, 找到单进程饱和点 (首遍引擎锁串行化的上限);
  - 多实例: N 个网关进程 (每进程独立加载模型), 会话轮询分发到各实例,
    对比总吞吐/延迟是否随实例数线性改善。

配合 run_paraformer_suite.sh 使用时, 网关以 FINAL_ASR_BACKEND=mock 启动
(二遍 mock 立即返回), 从而把测量目标隔离在首遍链路上。

协议 (与 demo/e2e 客户端一致):
  Text {"type":"start",...} -> session_ready -> Binary PCM 分帧推流 ->
  {"type":"commit"} + {"type":"stop"} -> session_finished

用法:
  python3 benchmark/bench_gateway.py \
      --uris ws://127.0.0.1:21001,ws://127.0.0.1:21002 \
      --sessions 1,4,8,16 --audio fixtures/sample_16k.wav

输出 (写入 --out_dir):
  gateway_ws.csv    每个 (实例数 x 并发会话数) 一行的核心指标
  gateway_details.json  每会话明细
  bench_gateway.log 完整过程日志
"""

import os
import sys
import json
import time
import argparse
import asyncio
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import websockets

from benchmark.bench_common import (
    REPO_ROOT,
    TARGET_SR,
    load_audio_pcm16,
    audio_duration_sec,
    slice_segment,
    parse_int_list,
    percentile_stats,
    setup_logger,
    new_run_dir,
    CsvWriter,
)

GATEWAY_CSV_FIELDS = [
    "run_id", "instances", "sessions", "rate", "stream_sec", "chunk_ms", "enable_spk",
    "success", "failed",
    "first_partial_ms_p50", "first_partial_ms_p90", "first_partial_ms_p95", "first_partial_ms_avg", "first_partial_ms_max",
    "end_to_end_ms_p50", "end_to_end_ms_p90", "end_to_end_ms_p95", "end_to_end_ms_avg",
    "wall_sec", "total_audio_sec", "rtx",
    "partials", "provisionals", "finals", "fallbacks", "chars", "errors",
    "per_instance",
]


class GatewaySession:
    """单路会话: 真实音频按 rate 倍速推流, 统计首帧/收尾延迟与消息计数。"""

    def __init__(self, idx: int, uri: str, pcm: bytes, args):
        self.idx = idx
        self.uri = uri
        self.audio = slice_segment(pcm, (idx * args.stream_sec * 0.7) % audio_duration_sec(pcm), args.stream_sec)
        self.args = args
        self.session_id = f"gwbench-{int(time.time() * 1000)}-{idx}"
        # 指标
        self.success = False
        self.error: Optional[str] = None
        self.first_partial_ms: Optional[float] = None
        self.end_to_end_ms: Optional[float] = None
        self.partials = 0
        self.provisionals = 0
        self.finals = 0
        self.fallbacks = 0
        self.chars = 0
        self.finished = asyncio.Event()
        self._ready = False
        self._t_first_partial: Optional[float] = None  # 首个非空 partial 到达时刻

    async def run(self) -> "GatewaySession":
        args = self.args
        chunk_size = int(TARGET_SR * 2 * args.chunk_ms / 1000)
        t_first_sent: Optional[float] = None
        try:
            async with websockets.connect(self.uri, max_size=10 * 1024 * 1024) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "type": "start",
                            "session_id": self.session_id,
                            "language": "zh",
                            "enable_spk": args.enable_spk,
                            "hotwords": args.hotwords.split(",") if args.hotwords else [],
                        }
                    )
                )
                recv_task = asyncio.create_task(self._receiver(ws))
                try:
                    # 等待 session_ready (最多 10s)
                    deadline = time.perf_counter() + 10.0
                    while not self._ready and time.perf_counter() < deadline:
                        await asyncio.sleep(0.05)
                    if not self._ready:
                        raise RuntimeError("no session_ready within 10s")

                    t0 = time.perf_counter()
                    for off in range(0, len(self.audio), chunk_size):
                        if self.finished.is_set():
                            break
                        if off == 0:
                            t_first_sent = time.perf_counter()
                        await ws.send(self.audio[off : off + chunk_size])
                        ahead = (off + chunk_size) / (TARGET_SR * 2) / args.rate - (time.perf_counter() - t0)
                        if ahead > 0:
                            await asyncio.sleep(ahead)

                    await ws.send(json.dumps({"type": "commit"}))
                    await ws.send(json.dumps({"type": "stop"}))
                    await asyncio.wait_for(self.finished.wait(), timeout=args.finish_timeout)
                    if self._t_first_partial is not None and t_first_sent is not None:
                        # 首字延迟 = 首个非空 partial 到达 - 首帧音频发出
                        self.first_partial_ms = (self._t_first_partial - t_first_sent) * 1000
                    if t_first_sent is not None:
                        self.end_to_end_ms = (time.perf_counter() - t_first_sent) * 1000
                    self.success = True
                finally:
                    recv_task.cancel()
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)[:300]
        return self

    async def _receiver(self, ws):
        try:
            async for raw in ws:
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "session_ready":
                    self._ready = True
                elif mtype == "partial":
                    self.partials += 1
                    if msg.get("text") and self._t_first_partial is None:
                        self._t_first_partial = time.perf_counter()
                elif mtype == "provisional":
                    self.provisionals += 1
                    self.chars += len(msg.get("text") or "")
                elif mtype == "final":
                    self.finals += 1
                    if msg.get("final_source") != "qwen3-asr":
                        self.fallbacks += 1
                    self.chars += len(msg.get("text") or "")
                elif mtype == "error":
                    self.error = str(msg.get("message") or msg)[:300]
                elif mtype == "session_finished":
                    self.finished.set()
                    return
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            pass


async def run_level(uris: List[str], sessions: int, pcm: bytes, args, logger) -> dict:
    """一个并发级别: sessions 路会话同时开跑, 轮询分发到各实例。"""
    per_instance: Dict[str, dict] = {u: {"sessions": 0, "success": 0} for u in uris}
    session_objs = []
    for i in range(sessions):
        uri = uris[i % len(uris)]
        s = GatewaySession(i, uri, pcm, args)
        session_objs.append((uri, s))

    async def launch(i_uri_s):
        idx, (uri, s) = i_uri_s
        if idx > 0 and args.stagger_ms > 0:  # 错峰建连, 避免瞬时 accept 风暴
            await asyncio.sleep(args.stagger_ms * (idx % len(uris)) / 1000.0)
        per_instance[uri]["sessions"] += 1
        await s.run()
        if s.success:
            per_instance[uri]["success"] += 1

    t_wall = time.perf_counter()
    await asyncio.gather(*[launch((i, x)) for i, x in enumerate(session_objs)])
    wall = time.perf_counter() - t_wall

    fp_ms = [s.first_partial_ms for _, s in session_objs if s.first_partial_ms is not None]
    e2e_ms = [s.end_to_end_ms for _, s in session_objs if s.end_to_end_ms is not None]

    return {
        "sessions": sessions,
        "success": sum(1 for _, s in session_objs if s.success),
        "failed": sessions - sum(1 for _, s in session_objs if s.success),
        "first_partial": percentile_stats([m / 1000.0 for m in fp_ms]),
        "end_to_end": percentile_stats([m / 1000.0 for m in e2e_ms]),
        "wall_sec": round(wall, 2),
        "total_audio_sec": round(sum(len(s.audio) / (TARGET_SR * 2) for _, s in session_objs), 1),
        "partials": sum(s.partials for _, s in session_objs),
        "provisionals": sum(s.provisionals for _, s in session_objs),
        "finals": sum(s.finals for _, s in session_objs),
        "fallbacks": sum(s.fallbacks for _, s in session_objs),
        "chars": sum(s.chars for _, s in session_objs),
        "errors": sum(1 for _, s in session_objs if s.error),
        "per_instance": per_instance,
        "session_details": [
            {
                "idx": s.idx,
                "uri": uri,
                "success": s.success,
                "first_partial_ms": s.first_partial_ms,
                "end_to_end_ms": s.end_to_end_ms,
                "partials": s.partials,
                "provisionals": s.provisionals,
                "finals": s.finals,
                "fallbacks": s.fallbacks,
                "chars": s.chars,
                "error": s.error,
            }
            for uri, s in session_objs
        ],
    }


async def async_main(args, logger):
    uris = [u.strip() for u in args.uris.split(",") if u.strip()]
    pcm = load_audio_pcm16(args.audio)
    logger.info(
        f"网关实例: {uris} | 音频 {args.audio} ({audio_duration_sec(pcm):.0f}s) | "
        f"每会话取 {args.stream_sec}s @ {args.rate}x 推流 | spk={'on' if args.enable_spk else 'off'}"
    )

    # 连通性预检: 逐实例起一个空会话并干净收尾
    for uri in uris:
        try:
            async with websockets.connect(uri, open_timeout=5) as ws:
                await ws.send(
                    json.dumps({"type": "start", "session_id": f"probe-{int(time.time())}", "language": "zh", "enable_spk": False})
                )
                for _ in range(10):  # 最多等 10 条消息内出现 ready/finished
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                    mtype = json.loads(raw).get("type")
                    if mtype == "session_ready":
                        break
                    if mtype == "error":
                        raise RuntimeError(raw)
                else:
                    raise RuntimeError("no session_ready")
                await ws.send(json.dumps({"type": "stop"}))
                try:
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=10)
                        if json.loads(raw).get("type") in ("session_finished", "error"):
                            break
                except (asyncio.TimeoutError, websockets.ConnectionClosed):
                    pass
            logger.info(f"[预检] {uri} OK")
        except Exception as e:  # noqa: BLE001
            logger.error(f"[预检] {uri} 不可用: {e!r}")
            sys.exit(3)

    # 预热: 每实例跑短会话跑通全链路 (首次推理的内存分配/JIT/线程池唤醒),
    # 指标不进入统计, 避免第一级会话的 first_partial 被冷启动污染
    if args.warmup_sessions > 0:
        warm_args = argparse.Namespace(**vars(args))
        warm_args.stream_sec = args.warmup_sec
        for rnd in range(args.warmup_sessions):
            tasks = [GatewaySession(-(rnd + 1), uri, pcm, warm_args).run() for uri in uris]
            done = await asyncio.gather(*tasks)
            n_ok = sum(1 for s in done if s.success)
            logger.info(f"[预热] 第 {rnd + 1}/{args.warmup_sessions} 轮: {n_ok}/{len(done)} 实例短会话完成")
            if n_ok < len(done):
                logger.error(
                    f"[预热] 失败会话: {[s.error for s in done if not s.success]} —— 服务异常, 中止压测"
                )
                sys.exit(3)
            if args.cooldown > 0:
                await asyncio.sleep(args.cooldown)

    csv_path = os.path.join(args.out_dir, "gateway_ws.csv")
    writer = CsvWriter(csv_path, GATEWAY_CSV_FIELDS, logger)
    all_details = []

    for sessions in args.sessions_list:
        run_id = f"inst{len(uris)}_sess{sessions}"
        logger.info(f"===== [{run_id}] {len(uris)} 实例 x {sessions} 并发会话 ...")
        res = await run_level(uris, sessions, pcm, args, logger)
        fp, e2e = res["first_partial"], res["end_to_end"]
        row = {
            "run_id": run_id,
            "instances": len(uris),
            "sessions": sessions,
            "rate": args.rate,
            "stream_sec": args.stream_sec,
            "chunk_ms": args.chunk_ms,
            "enable_spk": args.enable_spk,
            "success": res["success"],
            "failed": res["failed"],
            "first_partial_ms_p50": round(fp.get("p50_ms", 0.0), 1),
            "first_partial_ms_p90": round(fp.get("p90_ms", 0.0), 1),
            "first_partial_ms_p95": round(fp.get("p95_ms", 0.0), 1),
            "first_partial_ms_avg": round(fp.get("avg_ms", 0.0), 1),
            "first_partial_ms_max": round(fp.get("max_ms", 0.0), 1),
            "end_to_end_ms_p50": round(e2e.get("p50_ms", 0.0), 1),
            "end_to_end_ms_p90": round(e2e.get("p90_ms", 0.0), 1),
            "end_to_end_ms_p95": round(e2e.get("p95_ms", 0.0), 1),
            "end_to_end_ms_avg": round(e2e.get("avg_ms", 0.0), 1),
            "wall_sec": res["wall_sec"],
            "total_audio_sec": res["total_audio_sec"],
            "rtx": round(res["total_audio_sec"] / res["wall_sec"], 2) if res["wall_sec"] else 0.0,
            "partials": res["partials"],
            "provisionals": res["provisionals"],
            "finals": res["finals"],
            "fallbacks": res["fallbacks"],
            "chars": res["chars"],
            "errors": res["errors"],
            "per_instance": json.dumps(res["per_instance"], ensure_ascii=False),
        }
        writer.write_row(row)
        all_details.append({"run_id": run_id, **{k: v for k, v in res.items() if k not in ("first_partial", "end_to_end")}})
        logger.info(
            f"  -> 成功 {res['success']}/{sessions} | 首partial p50/p95 = "
            f"{row['first_partial_ms_p50']}/{row['first_partial_ms_p95']} ms | "
            f"端到端 p50 = {row['end_to_end_ms_p50']} ms | 墙钟 {res['wall_sec']}s | "
            f"rtx={row['rtx']} | err={res['errors']}"
        )
        if args.cooldown > 0 and sessions != args.sessions_list[-1]:
            logger.info(f"  冷却 {args.cooldown}s ...")
            await asyncio.sleep(args.cooldown)

    with open(os.path.join(args.out_dir, "gateway_details.json"), "w", encoding="utf-8") as f:
        json.dump(all_details, f, ensure_ascii=False, indent=1)
    logger.info(f"完成. 结果: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="网关级 WS 并发压测 (多实例轮询负载均衡)", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--uris", required=True, help="网关 WS 地址, 逗号分隔多个实例 (会话轮询分发)")
    parser.add_argument("--sessions", default="1,4,8,16", help="并发会话数列表")
    parser.add_argument("--audio", default=os.path.join(REPO_ROOT, "fixtures", "sample_16k.wav"))
    parser.add_argument("--stream_sec", type=float, default=30.0, help="每会话音频时长(s)")
    parser.add_argument("--rate", type=float, default=1.0, help="推流倍速 (1.0=真实语速)")
    parser.add_argument("--chunk_ms", type=int, default=60)
    parser.add_argument("--enable_spk", action="store_true", help="启用说话人 (默认关闭, 首遍隔离测试)")
    parser.add_argument("--warmup_sessions", type=int, default=1, help="每实例预热短会话轮数 (不计入统计)")
    parser.add_argument("--warmup_sec", type=float, default=5.0, help="预热会话音频时长(s)")
    parser.add_argument("--hotwords", default="", help="热词, 逗号分隔 (可空)")
    parser.add_argument("--stagger_ms", type=int, default=200, help="同实例会话建连错峰(ms)")
    parser.add_argument("--finish_timeout", type=float, default=90.0, help="stop 后等待 session_finished 超时(s)")
    parser.add_argument("--cooldown", type=float, default=8.0, help="级别间冷却(s)")
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "benchmark", "results"))
    args = parser.parse_args()

    args.sessions_list = parse_int_list(args.sessions)
    run_dir = new_run_dir(
        args.out_dir,
        f"gateway_i{len([u for u in args.uris.split(',') if u.strip()])}",
        extra_meta={"tool": "bench_gateway.py", "uris": args.uris, "audio": args.audio, "cli": vars(args)},
    )
    args.out_dir = run_dir
    logger = setup_logger(os.path.join(run_dir, "bench_gateway.log"))
    asyncio.run(async_main(args, logger))


if __name__ == "__main__":
    main()
