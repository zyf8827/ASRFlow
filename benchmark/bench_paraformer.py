#!/usr/bin/env python3
"""
首遍 Paraformer 流式模型 CPU 推理压测 (引擎级, 绕过 WebSocket/网关)。

目的: 单独量化 Pass-1 模型本身在 CPU 上的性能, 回答两个问题:
  1. 单进程能承载多少路实时流? (引擎 process_chunk 内部全局锁串行化推理,
     单进程并发流共享一次推理 —— 本脚本用锁等待时间直接量化排队)
  2. 多进程 (每进程独立加载模型) 负载均衡下, 吞吐是否随进程数线性增长?

测试矩阵: pacing(offline/realtime) x procs(进程数) x streams(每进程并发流数)
  - offline : 不限速尽快推流, 测纯算力吞吐 (audio_sec / 墙钟耗时 = 实时倍率)
  - realtime: 按真实语速推流 (chunk 间隔=chunk_ms), 测实时承载能力与排队延迟

用法 (由 run_paraformer_suite.sh 编排, 也可单独执行):
  python3 benchmark/bench_paraformer.py \
      --config config/config.default.yaml --audio fixtures/sample_16k.wav \
      --pacing offline,realtime --procs 1,2,4 --streams 1,4

输出 (写入 --out_dir):
  paraformer_engine.csv   每个测试组合一行的核心指标
  paraformer_scaling.csv  多进程扩展性 (吞吐/加速比/并行效率, 以 procs=1 为基线)
  bench_paraformer.log    完整过程日志
  details_*.json          --save_details 时逐 chunk 延迟明细
"""

import os
import sys
import json
import time
import argparse
import threading
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.bench_common import (
    REPO_ROOT,
    TARGET_SR,
    audio_duration_sec,
    load_audio_pcm16,
    parse_int_list,
    percentile_stats,
    setup_logger,
    new_run_dir,
    CsvWriter,
)

ENGINE_CSV_FIELDS = [
    "run_id", "pacing", "procs", "streams_per_proc", "total_streams",
    "threads_per_proc", "chunk_ms", "stream_sec",
    "total_audio_sec", "wall_sec", "throughput_rtx", "compute_rtf",
    "call_ms_p50", "call_ms_p90", "call_ms_p95", "call_ms_p99", "call_ms_avg", "call_ms_max",
    "wait_ms_p50", "wait_ms_p90", "wait_ms_p95", "wait_ms_p99", "wait_ms_avg", "wait_ms_max",
    "lag_ms_p50", "lag_ms_p95", "lag_ms_max", "late_ratio",
    "chunks", "chars_total", "errors",
]
SCALING_CSV_FIELDS = [
    "run_id", "pacing", "streams_per_proc", "procs", "threads_per_proc",
    "throughput_rtx", "speedup_vs_1proc", "parallel_efficiency", "linear",
]


class _TimingLock:
    """代理推理锁, 记录每次 acquire 的排队等待时间 (备用诊断工具)。

    引擎 process_chunk 内部 `with self._lock:` 串行化推理; 用该代理替换后,
    wait=锁排队时间, call-wait≈纯推理时间, 从而把 排队 与 计算 分开量化。
    (streams=1 时 wait≈0, call 即无竞争纯推理延迟, 作为单路基线)
    """

    def __init__(self, inner: threading.Lock):
        self._inner = inner
        self._local = threading.local()

    def acquire(self, *args, **kwargs):
        t0 = time.perf_counter()
        self._inner.acquire(*args, **kwargs)
        self._local.wait_sec = time.perf_counter() - t0

    def release(self):
        self._inner.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    @property
    def last_wait_sec(self) -> float:
        return getattr(self._local, "wait_sec", 0.0)


def _worker_main(worker_args: dict, audio_path: str, result_queue: mp.Queue, start_event):
    """子进程: 独立加载 ONNX 引擎, 跑 streams 路并发流, 回传指标。

    注意: spawn 启动, 本函数内 import 顺序敏感 —— 先限制线程数再加载依赖。
    """
    wid = worker_args["wid"]
    try:
        # 1) 线程限制 (必须在 import torch 之前生效)
        threads = worker_args["threads"]
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(threads)
        import torch

        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # 已初始化并行环境时忽略

        # 2) 加载配置与引擎 (streaming_asr 段, 设备固定 cpu)
        from config.settings import AppConfig

        cfg = AppConfig.load(worker_args["config_path"])
        cfg.streaming_asr.device = "cpu"
        from core.streaming_asr.onnx_batched_streaming import OnnxBatchedStreamingEngine

        t_load = time.perf_counter()
        cfg.streaming_asr.onnx_model_dir = worker_args.get("onnx_dir", "models/onnx")
        cfg.streaming_asr.onnx_quantize = worker_args.get("onnx_quant", True)
        cfg.streaming_asr.batch_size = worker_args.get("batch_size", 8)
        cfg.streaming_asr.batch_window_ms = worker_args.get("batch_window_ms", 15)
        cfg.streaming_asr.onnx_intra_op_threads = worker_args.get("onnx_threads", threads)
        engine = OnnxBatchedStreamingEngine(cfg.streaming_asr)
        # 排队等待记录在 engine.last_wait_sec (thread-local)
        load_sec = time.perf_counter() - t_load

        # 3) 解码音频并预热 (触发懒初始化, 避免首 chunk 延迟污染统计)
        pcm = load_audio_pcm16(audio_path)
        warmup_sec = min(worker_args["warmup_sec"], max(1.0, audio_duration_sec(pcm) - 1))
        warm_pcm = pcm[: int(warmup_sec * TARGET_SR * 2)]
        chunk_bytes = int(TARGET_SR * 2 * worker_args["chunk_ms"] / 1000)
        cache: dict = {}
        t_warm = time.perf_counter()
        for off in range(0, len(warm_pcm), chunk_bytes):
            engine.process_chunk(warm_pcm[off : off + chunk_bytes], cache)
        engine.flush(cache)
        warm_cost = time.perf_counter() - t_warm

        result_queue.put(
            ("ready", {"wid": wid, "load_sec": round(load_sec, 2), "warmup_sec_cost": round(warm_cost, 2)})
        )
        start_event.wait(timeout=600)

        # 4) 并发推 streams 路流 (线程模型对齐网关: 引擎被多线程共享, 锁串行化)
        pacing = worker_args["pacing"]
        stream_sec = worker_args["stream_sec"]
        n_streams = worker_args["streams"]
        audio_total = len(pcm)

        def run_stream(sid: int, stats: dict, lat_call: list, lat_wait: list, lat_lag: list):
            # 每路流取不同起始偏移, 避免所有流喂完全相同的音频
            seg_start = int(((wid * n_streams + sid) * stream_sec) * TARGET_SR * 2) % audio_total
            nbytes = int(stream_sec * TARGET_SR * 2)
            seg = (
                pcm[seg_start : seg_start + nbytes]
                if seg_start + nbytes <= audio_total
                else (pcm[seg_start:] + pcm[: nbytes - (audio_total - seg_start)])
            )
            cache = {}
            chars = 0
            t_stream = time.perf_counter()
            n_chunks = 0

            def _read_wait():
                # onnx 引擎的排队等待记录在 engine.last_wait_sec (thread-local)
                return engine.last_wait_sec

            for off in range(0, len(seg) - chunk_bytes + 1, chunk_bytes):
                # realtime: chunk k 的真实到达时刻 = t_stream + 音频偏移, 睡到该时刻再发
                scheduled_elapsed = off / (TARGET_SR * 2) if pacing == "realtime" else None
                if scheduled_elapsed is not None:
                    ahead = scheduled_elapsed - (time.perf_counter() - t_stream)
                    if ahead > 0:
                        time.sleep(ahead)
                t0 = time.perf_counter()
                text = engine.process_chunk(seg[off : off + chunk_bytes], cache)
                t1 = time.perf_counter()
                lat_call.append(t1 - t0)
                lat_wait.append(_read_wait())
                if scheduled_elapsed is not None:
                    lat_lag.append(max(0.0, (t0 - t_stream) - scheduled_elapsed))
                chars += len(text)
                n_chunks += 1
            t_flush0 = time.perf_counter()
            tail = engine.flush(cache)
            lat_call.append(time.perf_counter() - t_flush0)
            lat_wait.append(_read_wait())
            chars += len(tail)
            with stats["lock"]:
                stats["audio_sec"] += len(seg) / (TARGET_SR * 2)  # PCM16: 2 字节/样本
                stats["chars"] += chars
                stats["chunks"] += n_chunks + 1
                stats["wall_sec"] = max(stats["wall_sec"], time.perf_counter() - t_stream)

        stats = {"lock": threading.Lock(), "audio_sec": 0.0, "chars": 0, "chunks": 0, "wall_sec": 0.0}
        lat_call: list = []
        lat_wait: list = []
        lat_lag: list = []
        threads_list = [
            threading.Thread(target=run_stream, args=(s, stats, lat_call, lat_wait, lat_lag))
            for s in range(n_streams)
        ]
        t_begin = time.perf_counter()
        for t in threads_list:
            t.start()
        for t in threads_list:
            t.join()
        elapsed = time.perf_counter() - t_begin

        result_queue.put(
            (
                "done",
                {
                    "wid": wid,
                    "audio_sec": stats["audio_sec"],
                    "chars": stats["chars"],
                    "chunks": stats["chunks"],
                    "wall_sec": elapsed,
                    "compute_sec": sum(lat_call),
                    "lat_call_raw": lat_call,
                    "lat_wait_raw": lat_wait,
                    "lat_lag_raw": lat_lag,
                    "errors": [],
                },
            )
        )
    except Exception as e:  # noqa: BLE001 - 子进程任何异常都要回传, 否则父进程永久等待
        result_queue.put(("error", {"wid": wid, "error": repr(e)}))


def run_combo(args, pacing: str, procs: int, streams: int, run_id: str, logger) -> dict:
    """跑一个 (pacing, procs, streams) 组合并聚合指标。"""
    threads = args.threads if args.threads > 0 else max(1, (os.cpu_count() or 1) // procs)
    logger.info(
        f"===== [{run_id}] pacing={pacing} procs={procs} streams/proc={streams} "
        f"threads/proc={threads} (共 {procs * streams} 路, 每路 {args.stream_sec}s 音频)"
    )

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    start_event = ctx.Event()
    workers = []
    try:
        for wid in range(procs):
            worker_args = {
                "wid": wid,
                "config_path": args.config,
                "threads": threads,
                "pacing": pacing,
                "streams": streams,
                "stream_sec": args.stream_sec,
                "chunk_ms": args.chunk_ms,
                "warmup_sec": args.warmup_sec,
                "onnx_dir": args.onnx_dir,
                "onnx_quant": not args.onnx_fp32,
                "batch_size": args.batch_size,
                "batch_window_ms": args.batch_window_ms,
            }
            p = ctx.Process(target=_worker_main, args=(worker_args, args.audio, result_queue, start_event))
            p.start()
            workers.append(p)

        def pump_message(timeout_sec: float):
            """取一条 worker 消息; 超时且所有 worker 已死时给出诊断而非卡死。"""
            try:
                return result_queue.get(timeout=min(30.0, timeout_sec))
            except Exception:
                alive = [i for i, p in enumerate(workers) if p.is_alive()]
                if alive:
                    return None  # 仍在跑, 外层按 deadline 继续等
                raise RuntimeError(
                    f"所有 worker 进程已退出但未回传结果 (exitcode={workers[0].exitcode}), "
                    f"可能被 OOM killer 杀死, 已收到 {len(results)}/{procs} 份结果"
                )

        # 等全部就绪 (模型加载+预热完成); 任一失败则中止本组合
        results = []
        n_ready = 0
        deadline = time.time() + args.load_timeout
        while n_ready < procs:
            msg = pump_message(deadline - time.time())
            if msg is None:
                if time.time() > deadline:
                    raise RuntimeError(f"等待 worker 就绪超时 (> {args.load_timeout}s), 已就绪 {n_ready}/{procs}")
                continue
            kind, payload = msg
            if kind == "ready":
                n_ready += 1
                logger.info(
                    f"  worker-{payload['wid']} 就绪 (模型加载 {payload['load_sec']}s, "
                    f"预热 {payload['warmup_sec_cost']}s)"
                )
            elif kind == "error":
                raise RuntimeError(f"worker-{payload['wid']} 失败: {payload['error']}")
            elif kind == "done":
                results.append(payload)  # 提前完成(异常路径), 留待后面统一校验

        logger.info(f"  全部 {procs} 个 worker 就绪, 同步开跑 ...")
        start_event.set()
        t_wall = time.perf_counter()
        deadline = time.time() + args.run_timeout
        while len(results) < procs:
            msg = pump_message(deadline - time.time())
            if msg is None:
                if time.time() > deadline:
                    raise RuntimeError(f"运行超时 (> {args.run_timeout}s), 已收到 {len(results)}/{procs} 份结果")
                continue
            kind, payload = msg
            if kind == "done":
                results.append(payload)
            elif kind == "error":
                logger.error(f"  worker-{payload['wid']} 运行失败: {payload['error']}")
                raise RuntimeError(f"worker 运行失败: {payload['error']}")
        wall_sec = time.perf_counter() - t_wall
    finally:
        start_event.set()  # 确保异常路径不卡 worker
        for p in workers:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()

    # 聚合 (跨 worker 合并原始延迟序列后统一取分位数)
    total_audio = sum(r["audio_sec"] for r in results)
    all_call = [v for r in results for v in r["lat_call_raw"]]
    all_wait = [v for r in results for v in r["lat_wait_raw"]]
    all_lag = [v for r in results for v in r["lat_lag_raw"]]
    compute_sec = sum(r["compute_sec"] for r in results)
    chars = sum(r["chars"] for r in results)
    chunks = sum(r["chunks"] for r in results)

    call_stats = percentile_stats(all_call)
    wait_stats = percentile_stats(all_wait)
    lag_stats = percentile_stats(all_lag) if all_lag else {}
    late_ratio = (
        sum(1 for v in all_lag if v * 1000.0 > args.chunk_ms / 2.0) / len(all_lag) if all_lag else 0.0
    )

    row = {
        "run_id": run_id,
        "pacing": pacing,
        "procs": procs,
        "streams_per_proc": streams,
        "total_streams": procs * streams,
        "threads_per_proc": threads,
        "chunk_ms": args.chunk_ms,
        "stream_sec": args.stream_sec,
        "total_audio_sec": round(total_audio, 2),
        "wall_sec": round(wall_sec, 3),
        # 实时倍率: 处理的音频秒数 / 墙钟秒数 (>=1 才能承载对应路数的实时流)
        "throughput_rtx": round(total_audio / wall_sec, 3) if wall_sec > 0 else 0.0,
        # 纯算力 RTF 的倒数 (所有 worker 推理时间之和 vs 音频量)
        "compute_rtf": round(total_audio / compute_sec, 3) if compute_sec > 0 else 0.0,
        **{f"call_ms_{k.replace('_ms', '')}": round(v, 2) for k, v in call_stats.items() if k != "count"},
        **{f"wait_ms_{k.replace('_ms', '')}": round(v, 2) for k, v in wait_stats.items() if k != "count"},
        "lag_ms_p50": round(lag_stats.get("p50_ms", 0.0), 2),
        "lag_ms_p95": round(lag_stats.get("p95_ms", 0.0), 2),
        "lag_ms_max": round(lag_stats.get("max_ms", 0.0), 2),
        "late_ratio": round(late_ratio, 4),
        "chunks": chunks,
        "chars_total": chars,
        "errors": sum(len(r["errors"]) for r in results),
    }
    if chars == 0:
        logger.warning("  [SANITY] 本组合输出的识别字符数为 0, 请检查音频/模型是否匹配!")

    logger.info(
        f"  -> 音频 {total_audio:.1f}s / 墙钟 {wall_sec:.2f}s = {row['throughput_rtx']}x 实时 | "
        f"compute RTF(1/x)={row['compute_rtf']} | call p50/p95/p99 = "
        f"{row['call_ms_p50']}/{row['call_ms_p95']}/{row['call_ms_p99']} ms | "
        f"lock wait p95 = {row['wait_ms_p95']} ms"
    )
    if args.save_details:
        detail_path = os.path.join(args.out_dir, f"details_{run_id}.json")
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "row": row,
                    "per_chunk_call_ms": [round(v * 1000, 3) for v in all_call],
                    "per_chunk_wait_ms": [round(v * 1000, 3) for v in all_wait],
                    "per_worker": [
                        {k: r[k] for k in ("wid", "audio_sec", "chars", "chunks", "wall_sec", "compute_sec")}
                        for r in results
                    ],
                },
                f,
                ensure_ascii=False,
            )
    return row


def main():
    parser = argparse.ArgumentParser(
        description="首遍 Paraformer CPU 引擎级压测 (多进程扩展性)", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "config.default.yaml"))
    parser.add_argument("--audio", default=os.path.join(REPO_ROOT, "fixtures", "sample_16k.wav"), help="测试语音音频 (mp3/wav)")
    parser.add_argument("--pacing", default="offline,realtime", help="离线吞吐/实时节奏, 逗号分隔")
    parser.add_argument("--procs", default="1,2,4", help="进程数列表 (每进程独立加载模型)")
    parser.add_argument("--streams", default="1,4", help="每进程并发流数列表")
    parser.add_argument("--threads", type=int, default=0, help="每进程 torch 线程数 (0=auto: 核数/进程数)")
    parser.add_argument("--chunk_ms", type=int, default=60, help="每次 process_chunk 喂入的音频时长 (对齐网关推流帧)")
    parser.add_argument("--onnx_dir", default="models/onnx", help="ONNX 导出物目录")
    parser.add_argument("--onnx_fp32", action="store_true", help="使用 fp32 图 (默认 int8)")
    parser.add_argument("--batch_size", type=int, default=8, help="单次合批最大流数")
    parser.add_argument("--batch_window_ms", type=int, default=15, help="攒批等待窗口(ms)")
    parser.add_argument("--stream_sec", type=float, default=30.0, help="每路流音频时长(s)")
    parser.add_argument("--warmup_sec", type=float, default=5.0)
    parser.add_argument("--load_timeout", type=float, default=600.0, help="等待模型加载超时(s)")
    parser.add_argument("--run_timeout", type=float, default=1800.0, help="单组合运行超时(s)")
    parser.add_argument("--pause_sec", type=float, default=5.0, help="组合间冷却(s)")
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "benchmark", "results"))
    parser.add_argument("--save_details", action="store_true", help="保存逐 chunk 延迟明细 JSON")
    args = parser.parse_args()

    # 基线组合校验: 扩展性分析需要 procs=1 作基线
    pacings = [p.strip() for p in args.pacing.split(",") if p.strip()]
    procs_list = parse_int_list(args.procs)
    streams_list = parse_int_list(args.streams)

    run_dir = new_run_dir(
        args.out_dir,
        "paraformer_engine",
        extra_meta={
            "tool": "bench_paraformer.py",
            "config": args.config,
            "audio": args.audio,
            "matrix": {"pacing": pacings, "procs": procs_list, "streams": streams_list},
            "cli": vars(args),
        },
    )
    args.out_dir = run_dir  # details/日志等产物统一写入本次 run 目录
    logger = setup_logger(os.path.join(run_dir, "bench_paraformer.log"))

    # 打印生效的模型配置, 保证报告可溯源
    from config.settings import AppConfig

    cfg = AppConfig.load(args.config)
    sc = cfg.streaming_asr
    logger.info(
        f"引擎: OnnxBatchedStreamingEngine dir={args.onnx_dir} quant={not args.onnx_fp32} "
        f"batch={args.batch_size} window={args.batch_window_ms}ms chunk_size={sc.chunk_size}"
    )
    logger.info(f"音频: {args.audio}, 输出目录: {run_dir}")

    engine_csv = CsvWriter(os.path.join(run_dir, "paraformer_engine.csv"), ENGINE_CSV_FIELDS, logger)
    scaling_rows = []
    run_no = 0
    for pacing in pacings:
        for streams in streams_list:
            baseline_rtx = None
            for procs in procs_list:
                run_no += 1
                run_id = f"{pacing}_p{procs}s{streams}"
                try:
                    row = run_combo(args, pacing, procs, streams, run_id, logger)
                except Exception as e:  # noqa: BLE001 - 单组合失败不终止整个矩阵
                    logger.error(f"  [FAIL] 组合 {run_id} 失败: {e!r}, 跳过并继续")
                    time.sleep(args.pause_sec)
                    continue
                engine_csv.write_row(row)
                if procs == 1:
                    baseline_rtx = row["throughput_rtx"]
                elif baseline_rtx and baseline_rtx > 0:
                    speedup = row["throughput_rtx"] / baseline_rtx
                    scaling_rows.append(
                        {
                            "run_id": run_id,
                            "pacing": pacing,
                            "streams_per_proc": streams,
                            "procs": procs,
                            "threads_per_proc": row["threads_per_proc"],
                            "throughput_rtx": row["throughput_rtx"],
                            "speedup_vs_1proc": round(speedup, 3),
                            # 理想线性扩展时效率=1.0; >0.9 视为接近线性
                            "parallel_efficiency": round(speedup / procs, 3),
                            "linear": "yes" if speedup / procs >= 0.9 else "no",
                        }
                    )
                if args.pause_sec > 0 and not (pacing == pacings[-1] and procs == procs_list[-1] and streams == streams_list[-1]):
                    logger.info(f"  冷却 {args.pause_sec}s ...")
                    time.sleep(args.pause_sec)

    if scaling_rows:
        scaling_csv = CsvWriter(os.path.join(run_dir, "paraformer_scaling.csv"), SCALING_CSV_FIELDS, logger)
        for r in scaling_rows:
            scaling_csv.write_row(r)
        logger.info("多进程扩展性汇总 (基线 procs=1):")
        for r in scaling_rows:
            logger.info(
                f"  {r['pacing']} streams/proc={r['streams_per_proc']}: "
                f"{r['procs']} 进程 -> {r['throughput_rtx']}x 实时, "
                f"加速比 {r['speedup_vs_1proc']} ({r['parallel_efficiency']*100:.0f}% 线性)"
            )
    logger.info(f"完成. 结果: {run_dir}")


if __name__ == "__main__":
    main()
