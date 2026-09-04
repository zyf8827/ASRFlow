#!/usr/bin/env python3
"""
二遍 Qwen3-ASR (远程 vLLM, 单卡 4090) 并发压测 —— 直连 HTTP, 绕过本服务网关。

目的: 单独量化 Pass-2 模型服务的并发能力, 回答:
  1. 并发从 1 逐步增加时, 请求延迟如何变化 (批量收益 vs 排队)?
  2. 单 4090 的吞吐拐点/饱和并发是多少 (req/s 与 实时倍率 audio_sec/s)?
  3. 生产 8s 硬超时口径下, 高并发时超时率是多少? (--timeout 控制)

实现要点:
  - 复用 core.final_asr.qwen_engine.Qwen3ASREngine 发请求, HTTP 行为与生产完全一致
    (同样的 multipart /v1/audio/transcriptions、prompt 前缀、鉴权头);
  - 音频取自测试语音 (默认 fixtures/sample_16k.wav), 按指定时长切段并编码为 WAV;
  - 每个并发级别独立预热 + 固定请求数采样, 级别间冷却等待 GPU 队列排空。

用法 (由 run_qwen_suite.sh 编排, 也可单独执行):
  python3 benchmark/bench_qwen_vllm.py \
      --config config/config.default.yaml --audio fixtures/sample_16k.wav \
      --seg_sec_list 3,10 --concurrency 1,2,4,8 --requests_per_level 16 --timeout 30

输出 (写入 --out_dir):
  qwen_vllm.csv          每个并发级别一行的核心指标
  qwen_vllm_details.json 每请求明细 (延迟/字符数/错误), 便于报告画分布图
  bench_qwen_vllm.log    完整过程日志
"""

import os
import sys
import json
import time
import argparse
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.bench_common import (
    REPO_ROOT,
    load_audio_pcm16,
    audio_duration_sec,
    pcm_to_wav_bytes,
    parse_int_list,
    parse_float_list,
    percentile_stats,
    setup_logger,
    new_run_dir,
    CsvWriter,
)

QWEN_CSV_FIELDS = [
    "run_id", "seg_sec", "concurrency", "requests",
    "ok", "http_errors", "timeouts", "other_errors",
    "lat_ms_p50", "lat_ms_p90", "lat_ms_p95", "lat_ms_p99", "lat_ms_avg", "lat_ms_max",
    "req_per_sec", "audio_rtx",
    "chars_avg", "chars_min",
    "timeout_sec", "model", "url",
]


def classify_error(exc_name: str) -> str:
    """错误分类 (按异常类名): timeout / http / other, 与生产超时回退路径对齐。"""
    if "Timeout" in exc_name or "TimeoutError" in exc_name:
        return "timeouts"
    if "RuntimeError" in exc_name or "ClientError" in exc_name or "HTTP" in exc_name:
        return "http_errors"
    return "other_errors"


async def run_level(engine, wav_payloads, seg_sec: float, concurrency: int, logger) -> dict:
    """单个并发级别: semaphore 限并发, 全部请求完成后统计。"""
    sem = asyncio.Semaphore(concurrency)
    records = []

    async def one_request(idx: int):
        async with sem:
            t0 = time.perf_counter()
            try:
                text = await engine.transcribe(wav_payloads[idx % len(wav_payloads)])
                records.append({"i": idx, "ok": True, "ms": (time.perf_counter() - t0) * 1000, "chars": len(text)})
            except Exception as e:  # noqa: BLE001 - 单请求失败不中止整轮
                records.append(
                    {
                        "i": idx,
                        "ok": False,
                        "ms": (time.perf_counter() - t0) * 1000,
                        "error_type": type(e).__name__,
                        "error": repr(e)[:300],
                    }
                )

    t_wall = time.perf_counter()
    await asyncio.gather(*[one_request(i) for i in range(len(wav_payloads))])
    wall = time.perf_counter() - t_wall

    ok_ms = [r["ms"] for r in records if r["ok"]]
    ok_chars = [r["chars"] for r in records if r["ok"]]
    err_counts = {"http_errors": 0, "timeouts": 0, "other_errors": 0}
    for r in records:
        if not r["ok"]:
            err_counts[classify_error(r["error_type"])] += 1
    lat = percentile_stats([m / 1000.0 for m in ok_ms])

    return {
        "requests": len(records),
        "ok": len(ok_ms),
        **err_counts,
        "lat_ms": {k: round(v, 1) for k, v in lat.items() if k != "count"},
        "wall_sec": round(wall, 3),
        "req_per_sec": round(len(ok_ms) / wall, 3) if wall > 0 else 0.0,
        # 实时倍率: 成功请求的音频秒数 / 墙钟秒数 (衡量可支撑的二遍实时流数)
        "audio_rtx": round(len(ok_ms) * seg_sec / wall, 2) if wall > 0 else 0.0,
        "chars_avg": round(sum(ok_chars) / len(ok_chars), 1) if ok_chars else 0.0,
        "chars_min": min(ok_chars) if ok_chars else 0,
        "records": records,
    }


async def preflight(engine, pcm: bytes, logger) -> bool:
    """连通性预检: 发一个 1s 小请求, 失败则直接中止 (避免整轮压测白跑)。"""
    from benchmark.bench_common import slice_segment

    try:
        wav = pcm_to_wav_bytes(slice_segment(pcm, 0, 1.0))
        t0 = time.perf_counter()
        text = await engine.transcribe(wav)
        ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[预检] 连通正常: {ms:.0f} ms, 返回 {len(text)} 字符 ('{text[:20]}...')")
        return True
    except Exception as e:  # noqa: BLE001
        logger.error(f"[预检] 二遍服务不可用: {e!r} —— 请确认 vLLM 服务已启动且地址可达")
        return False


async def async_main(args, logger):
    from config.settings import AppConfig
    from core.final_asr.qwen_engine import Qwen3ASREngine
    from benchmark.bench_common import slice_segment

    cfg = AppConfig.load(args.config)
    fac = cfg.final_asr
    fac.vllm_url = args.url or fac.vllm_url
    fac.model_name = args.model or fac.model_name
    timeout = args.timeout if args.timeout > 0 else fac.hard_timeout_sec
    fac.hard_timeout_sec = timeout

    logger.info(f"二遍服务: {fac.vllm_url} model={fac.model_name} timeout={timeout}s prompt_prefix='{fac.prompt_prefix}'")

    # 构造引擎 (aiohttp session 级 total timeout, 与生产同机制)
    engine = Qwen3ASREngine(fac)

    pcm = load_audio_pcm16(args.audio)
    total_sec = audio_duration_sec(pcm)
    logger.info(f"音频: {args.audio} ({total_sec:.0f}s), 切段时长: {args.seg_sec_list}")

    if not await preflight(engine, pcm, logger):
        sys.exit(3)

    run_dir = args.out_dir
    csv_path = os.path.join(run_dir, "qwen_vllm.csv")
    engine_csv = CsvWriter(csv_path, QWEN_CSV_FIELDS, logger)
    details = []

    run_no = 0
    for seg_sec in args.seg_sec_list:
        # 请求负载: 不同起始偏移 (0.7 倍段长步进) 循环取段, 保证内容有差异
        wav_payloads = [
            pcm_to_wav_bytes(slice_segment(pcm, i * seg_sec * 0.7, seg_sec))
            for i in range(args.requests_per_level)
        ]
        logger.info(f"===== 段长 {seg_sec}s: {len(wav_payloads)} 个请求负载 (~{len(wav_payloads[0]) / 1024:.0f}KB WAV/个)")

        for concurrency in args.concurrency:
            run_no += 1
            run_id = f"seg{seg_sec}_c{concurrency}"
            # 预热: 串行发 N 个请求 (不计入统计), 消除首请求冷启动/GPU 降频影响
            if args.warmup > 0:
                logger.info(f"  预热 {args.warmup} 个请求 (不计入统计) ...")
                for _ in range(args.warmup):
                    await engine.transcribe(wav_payloads[0])
            logger.info(f"----- [{run_id}] 并发 {concurrency}, 共 {len(wav_payloads)} 个请求 ...")
            res = await run_level(engine, wav_payloads, seg_sec, concurrency, logger)
            row = {
                "run_id": run_id,
                "seg_sec": seg_sec,
                "concurrency": concurrency,
                "requests": res["requests"],
                "ok": res["ok"],
                "http_errors": res["http_errors"],
                "timeouts": res["timeouts"],
                "other_errors": res["other_errors"],
                "lat_ms_p50": res["lat_ms"].get("p50_ms"),
                "lat_ms_p90": res["lat_ms"].get("p90_ms"),
                "lat_ms_p95": res["lat_ms"].get("p95_ms"),
                "lat_ms_p99": res["lat_ms"].get("p99_ms"),
                "lat_ms_avg": res["lat_ms"].get("avg_ms"),
                "lat_ms_max": res["lat_ms"].get("max_ms"),
                "req_per_sec": res["req_per_sec"],
                "audio_rtx": res["audio_rtx"],
                "chars_avg": res["chars_avg"],
                "chars_min": res["chars_min"],
                "timeout_sec": timeout,
                "model": fac.model_name,
                "url": fac.vllm_url,
            }
            engine_csv.write_row(row)
            details.append(
                {"seg_sec": seg_sec, "concurrency": concurrency, "wall_sec": res["wall_sec"], "records": res["records"]}
            )
            logger.info(
                f"  -> ok={res['ok']}/{res['requests']} err(http/timeout/other)="
                f"{res['http_errors']}/{res['timeouts']}/{res['other_errors']} | "
                f"延迟 p50/p95/p99 = {row['lat_ms_p50']}/{row['lat_ms_p95']}/{row['lat_ms_p99']} ms | "
                f"{res['req_per_sec']} req/s, {res['audio_rtx']}x 实时音频"
            )
            if args.cooldown > 0:
                logger.info(f"  冷却 {args.cooldown}s (等待 GPU 队列排空) ...")
                await asyncio.sleep(args.cooldown)

    with open(os.path.join(run_dir, "qwen_vllm_details.json"), "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=1)
    await engine.close()
    logger.info(f"完成. 结果: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="二遍 Qwen3-ASR 远程 vLLM 并发压测", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "config.default.yaml"))
    parser.add_argument("--url", default=None, help="vLLM 地址 (默认取 config final_asr.vllm_url)")
    parser.add_argument("--model", default=None, help="模型名 (默认取 config final_asr.model_name)")
    parser.add_argument("--timeout", type=float, default=0, help="单请求超时(s), 0=用配置值; 压容量曲线建议 30s+, 生产口径 8s")
    parser.add_argument("--audio", default=os.path.join(REPO_ROOT, "fixtures", "sample_16k.wav"))
    parser.add_argument("--seg_sec_list", default="3,10", help="每段音频时长列表(s), 模拟 VAD 断句段长")
    parser.add_argument("--concurrency", default="1,2,4,8,16,24,32", help="并发级别列表")
    parser.add_argument("--requests_per_level", type=int, default=24, help="每个并发级别的总请求数")
    parser.add_argument("--warmup", type=int, default=2, help="每级别预热请求数")
    parser.add_argument("--cooldown", type=float, default=8.0, help="级别间冷却(s)")
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "benchmark", "results"))
    args = parser.parse_args()

    args.seg_sec_list = parse_float_list(args.seg_sec_list)
    args.concurrency = parse_int_list(args.concurrency)
    run_dir = new_run_dir(
        args.out_dir,
        "qwen_vllm",
        extra_meta={"tool": "bench_qwen_vllm.py", "config": args.config, "audio": args.audio, "cli": vars(args)},
    )
    args.out_dir = run_dir
    logger = setup_logger(os.path.join(run_dir, "bench_qwen_vllm.log"))
    asyncio.run(async_main(args, logger))


if __name__ == "__main__":
    main()
