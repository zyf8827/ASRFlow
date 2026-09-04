#!/usr/bin/env python3
"""
Benchmark 公共工具: 音频加载/切分、延迟统计、CSV 追加、运行目录/清单、日志初始化。

被以下压测脚本共用:
  - bench_paraformer.py  首遍 Paraformer CPU 引擎级压测 (多进程)
  - bench_qwen_vllm.py   二遍 Qwen3-ASR 远程 vLLM 并发压测
  - bench_gateway.py     网关级 WebSocket 多实例负载均衡压测
"""

import csv
import json
import os
import platform
import subprocess
import sys
import time
import datetime
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

TARGET_SR = 16000  # 协议固定 PCM16/16kHz/单声道


# ---------------------------------------------------------------------------
# 运行目录 / 清单 / 日志
# ---------------------------------------------------------------------------

def now_tag() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def host_info() -> Dict[str, object]:
    """采集主机信息写入 manifest, 报告引用时可直接溯源。"""
    info: Dict[str, object] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "logical_cores": os.cpu_count(),
        "cpu_model": "",
        "mem_total_gb": None,
    }
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["mem_total_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                    break
    except OSError:
        pass
    try:
        info["git_rev"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        info["git_rev"] = "unknown"
    return info


def new_run_dir(base_dir: str, name: str, extra_meta: Optional[Dict] = None) -> str:
    """创建 benchmark/results/<时间戳>_<name>/ 并写入 manifest.json。"""
    run_dir = os.path.join(base_dir, f"{now_tag()}_{name}")
    os.makedirs(run_dir, exist_ok=True)
    manifest = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "name": name,
        "host": host_info(),
    }
    if extra_meta:
        manifest.update(extra_meta)
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return run_dir


def setup_logger(log_file: str, level: str = "INFO"):
    """loguru 输出到 控制台 + 压测日志文件 (文件保留完整过程, 便于写报告回溯)。"""
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level=level)
    logger.add(log_file, level="DEBUG", encoding="utf-8", enqueue=False)
    return logger


# ---------------------------------------------------------------------------
# 音频
# ---------------------------------------------------------------------------

def load_audio_pcm16(path: str, max_sec: float = 0.0) -> bytes:
    """解码任意音频为 16kHz/mono/PCM16 字节流 (mp3 等压缩格式走 ffmpeg)。
    与 scripts/e2e_test.py 的解码方式保持一致。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"audio not found: {path}")
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", path,
            "-ac", "1", "-ar", str(TARGET_SR), "-sample_fmt", "s16",
            "-f", "s16le", "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    pcm = proc.stdout
    if max_sec and max_sec > 0:
        pcm = pcm[: int(TARGET_SR * 2 * max_sec)]
    return pcm


def audio_duration_sec(pcm: bytes) -> float:
    return len(pcm) / 2.0 / TARGET_SR


def slice_segment(pcm: bytes, offset_sec: float, seg_sec: float) -> bytes:
    """从 PCM 流中切 [offset_sec, offset_sec+seg_sec) 段, 越界自动回绕。"""
    total = len(pcm)
    start = int(offset_sec * TARGET_SR * 2) % total
    nbytes = int(seg_sec * TARGET_SR * 2)
    if start + nbytes <= total:
        return pcm[start : start + nbytes]
    # 回绕拼接 (循环音频)
    out = pcm[start:]
    while len(out) < nbytes:
        out += pcm[: nbytes - len(out)] if nbytes - len(out) <= total else pcm
    return out[:nbytes]


def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = TARGET_SR) -> bytes:
    """PCM16 打包为 WAV 容器 (二遍 vLLM transcriptions 接口要求)。"""
    from core.final_asr.qwen_engine import pcm_to_wav_bytes as _wrap

    return _wrap(pcm_bytes, sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# 统计与 CSV
# ---------------------------------------------------------------------------

def percentile_stats(values: List[float], unit_ms: bool = True) -> Dict[str, float]:
    """延迟分布: count/min/avg/p50/p90/p95/p99/max。输入为秒, 输出毫秒。"""
    if not values:
        keys = ("count", "min_ms", "avg_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms")
        return {k: 0.0 for k in keys}
    arr = np.asarray(values, dtype=np.float64)
    scale = 1000.0 if unit_ms else 1.0
    return {
        "count": int(arr.size),
        "min_ms": float(arr.min() * scale),
        "avg_ms": float(arr.mean() * scale),
        "p50_ms": float(np.percentile(arr, 50) * scale),
        "p90_ms": float(np.percentile(arr, 90) * scale),
        "p95_ms": float(np.percentile(arr, 95) * scale),
        "p99_ms": float(np.percentile(arr, 99) * scale),
        "max_ms": float(arr.max() * scale),
    }


class CsvWriter:
    """追加式 CSV: 首次写表头, 之后逐行追加并立即落盘 (中断也能保留已有数据)。"""

    def __init__(self, path: str, fieldnames: List[str], logger=None):
        self.path = path
        self.fieldnames = fieldnames
        self._logger = logger
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        if write_header:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        else:
            with open(path, "r", newline="", encoding="utf-8") as f:
                existing = next(csv.reader(f), [])
            if existing and existing != fieldnames and self._logger:
                self._logger.warning(
                    f"[CsvWriter] {path} 表头与本次字段不一致, 继续追加 (多余列留空)"
                )

    def write_row(self, row: Dict[str, object]):
        clean = {k: row.get(k, "") for k in self.fieldnames}
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(clean)
            f.flush()
            os.fsync(f.fileno())


def parse_int_list(text: str) -> List[int]:
    return [int(x) for x in str(text).replace(",", " ").split() if str(x).strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x) for x in str(text).replace(",", " ").split() if str(x).strip()]


def load_app_config(config_path: str):
    """加载完整配置 (默认 config/config.default.yaml), benchmark 以其为基准。"""
    from config.settings import AppConfig

    return AppConfig.load(config_path)
