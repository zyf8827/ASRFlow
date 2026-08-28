#!/usr/bin/env python3
"""
ASRFlow 实时转写 · 终端演示客户端（美观命令行输出）

效果：
  - 首遍 partial  : 当前句在底部活动行原位追加刷新（青色 ▶）
  - 首遍成句      : 活动行固化为一条"待定稿"行（暗色 ~），换行继续下一句
  - 二遍 final    : 回刷替换对应"待定稿"行（绿色 + 说话人 + 时间戳），不改写更早历史
  - 会话结束      : 汇总统计 + 可选完整转写（--full）

用法（配合服务端）:
    python3 demo/console_client.py --uri ws://127.0.0.1:10095 --audio fixtures/sample_16k.wav
    .venv/bin/python demo/console_client.py --rate 2.0 --max-sec 60 --hotwords 星巴克,电影院
    Mock 冒烟: 先以 mock 后端启动服务, 再 --uri ws://127.0.0.1:10195 --rate 50 --max-sec 60
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import unicodedata

import websockets

TARGET_SR = 16000


# ---------------------------------------------------------------- 流式性能指标


class StreamMetrics:
    """
    客户端侧性能测量。

    核心量 lag(t) = 推流墙钟时间 - 已识别音频位置(partial 的 end_ms):
      - lag 恒定  : 服务端跟得上当前倍速, 给出 RTF 上界 (RTF <= 1/实际倍速)
      - lag 线性增长: 服务端积压, 斜率 slope 反推服务端 RTF = R/(R-slope)
    另测: 首个 partial 延迟、二遍 final 延迟(provisional→final 同句间隔)。
    """

    def __init__(self):
        self.t0 = None
        self.audio_sec = 0.0
        self.stream_end = None
        self.partial_samples = []   # (墙钟相对 t0, 已识别音频秒)
        self.first_partial_delay = None
        self.prov_times = {}        # sentence_id -> 墙钟相对 t0
        self.final_latencies = []

    @property
    def lag_now(self) -> float:
        if not self.partial_samples:
            return 0.0
        w, e = self.partial_samples[-1]
        return w - e

    def mark_start(self):
        self.t0 = time.time()

    def mark_audio(self, sec: float):
        self.audio_sec = sec

    def mark_stream_end(self):
        self.stream_end = time.time()

    def on_partial(self, end_ms: int):
        if self.t0 is None:
            return
        w = time.time() - self.t0
        e = end_ms / 1000.0
        self.partial_samples.append((w, e))
        if self.first_partial_delay is None:
            self.first_partial_delay = w - e

    def on_provisional(self, sentence_id: int):
        if self.t0 is not None:
            self.prov_times[sentence_id] = time.time() - self.t0

    def on_final(self, sentence_id: int):
        p = self.prov_times.pop(sentence_id, None)
        if p is not None and self.t0 is not None:
            self.final_latencies.append((time.time() - self.t0) - p)

    @staticmethod
    def _slope(pts) -> float:
        n = len(pts)
        if n < 5:
            return 0.0
        mx = sum(x for x, _ in pts) / n
        my = sum(y for _, y in pts) / n
        sxx = sum((x - mx) ** 2 for x, _ in pts)
        if sxx < 1e-9:
            return 0.0
        return sum((x - mx) * (y - my) for x, y in pts) / sxx

    def summarize(self) -> list:
        """返回 [(label, value_str)] 行, 供彩色/纯文本两种渲染。"""
        out = []
        if self.t0 is None:
            return out
        wall = (self.stream_end or time.time()) - self.t0
        rate = self.audio_sec / wall if wall > 0 else 0.0
        out.append(("音频/推流", f"{self.audio_sec:.1f}s / {wall:.1f}s (实际倍速 {rate:.2f}x)"))

        if self.first_partial_delay is not None:
            out.append(("首个 partial 延迟", f"{max(0.0, self.first_partial_delay):.2f}s"))

        lags = [w - e for w, e in self.partial_samples]

        def fmt_lag(v: float) -> str:
            return f"领先 {-v:.2f}s" if v < 0 else f"{v:.2f}s"

        if lags:
            warm = [(w, w - e) for w, e in self.partial_samples if w >= min(3.0, wall / 3)]
            slope = self._slope(warm)
            if rate > 1.1:
                # 超实时推流: "滞后"为负(识别领先墙钟), 展示识别进度更直观
                e_last = self.partial_samples[-1][1]
                pct = 100.0 * e_last / max(self.audio_sec, 0.01)
                out.append((
                    "识别进度",
                    f"推流结束时已识别 {e_last:.1f}/{self.audio_sec:.1f}s ({pct:.0f}%)",
                ))
            else:
                lag_tail = sum(lags[-10:]) / min(10, len(lags))
                out.append((
                    "实时滞后 lag",
                    f"稳态 {fmt_lag(lag_tail)} / 最大 {fmt_lag(max(lags))}"
                    + (f" (以 {slope:.2f}s/s 增长)" if slope > 0.05 else " (稳定)"),
                ))
            if slope > 0.05 and rate > slope:
                out.append(("服务端 RTF 估算", f"≈ {rate / (rate - slope):.2f} (滞后增长, 积压态)"))
            elif rate > 0:
                out.append(("服务端 RTF 上界", f"≤ {1.0 / rate:.2f} (滞后稳定, 未积压)"))

        if self.final_latencies:
            fs = sorted(self.final_latencies)
            p95 = fs[max(0, int(len(fs) * 0.95) - 1)]
            out.append(("二遍 final 延迟", f"平均 {sum(fs)/len(fs):.2f}s / P95 {p95:.2f}s ({len(fs)} 句)"))
        return out

# ---------------------------------------------------------------- ANSI 终端渲染


def _cjk_width(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1


def display_width(s: str) -> int:
    return sum(_cjk_width(c) for c in s)


def wrap_display(text: str, width: int) -> list:
    """按显示宽度折行(ANSI 转义不计宽), 返回物理行列表。"""
    plain_budget = width
    lines, cur, cur_w = [], [], 0
    i = 0
    while i < len(text):
        # 透传 ANSI 转义序列
        if text[i] == "\x1b":
            j = text.find("m", i)
            if j == -1:
                j = len(text) - 1
            cur.append(text[i : j + 1])
            i = j + 1
            continue
        ch = text[i]
        w = _cjk_width(ch)
        if cur_w + w > plain_budget and cur:
            lines.append("".join(cur))
            cur, cur_w = [], 0
        cur.append(ch)
        cur_w += w
        i += 1
    if cur:
        lines.append("".join(cur))
    return lines or [""]


class Palette:
    def __init__(self, enabled: bool):
        self.on = enabled

    def _c(self, code: str, s: str) -> str:
        return f"\x1b[{code}m{s}\x1b[0m" if self.on else s

    def dim(self, s):    return self._c("2", s)
    def bold(self, s):   return self._c("1", s)
    def cyan(self, s):   return self._c("36", s)
    def green(self, s):  return self._c("32", s)
    def yellow(self, s): return self._c("33", s)
    def magenta(self, s): return self._c("35", s)
    def white(self, s):  return self._c("97", s)
    def grey(self, s):   return self._c("90", s)
    def red(self, s):    return self._c("31", s)


class TranscriptView:
    """
    终端转写渲染器。维护"已打印栈"(committed + pending 行)与底部"活动行"(当前 partial)。
    cursor 恒定停在全部已绘制内容的下一行行首; 通过相对光标移动实现 final 回刷替换。
    """

    def __init__(self, out, color: bool, metrics: "StreamMetrics"):
        self.out = out
        self.pal = Palette(color)
        self.tty = color
        self.metrics = metrics
        # stack: list of (sentence_id|None, [physical lines])
        self.stack = []          # 已固化行(历史, 按序)
        self.active_lines = []   # 活动 partial 块的物理行
        self.stats = {"partial": 0, "provisional": 0, "final": 0, "fallback": 0, "review": 0}

    # ---- 基础绘制 ----
    def _write(self, s: str):
        self.out.write(s)
        self.out.flush()

    def _width(self) -> int:
        try:
            cols = shutil.get_terminal_size((100, 24)).columns
        except Exception:
            cols = 100
        # 某些 pty/CI 环境上报 0 列, 钳制到可用下限
        return max(40, cols)

    def _fmt_ts(self, ms: int) -> str:
        s = max(0, int(ms // 1000))
        return f"{s // 60:02d}:{s % 60:02d}"

    def _render_row(self, text, speaker, start_ms, state, source=None, review=False) -> list:
        p = self.pal
        width = self._width()
        if state == "pending":
            head = p.cyan(p.dim("~")) + " "
            body = p.dim(p.white(text))
        else:
            tag = p.magenta(f"{speaker or 'SPK?':<4}") + p.grey("│")
            ts = p.grey(self._fmt_ts(start_ms)) + p.grey(" ")
            # offline 成功响应但判定无有效语音时, 定稿为空文本: 显示占位标记
            body_text = (
                p.dim("(空)")
                if not text
                else (p.green(text) if source == "qwen3-asr" else p.yellow(text))
            )
            marks = ""
            if source != "qwen3-asr":
                marks += " " + p.yellow("[首遍回退]")
            if review:
                marks += " " + p.grey("✱复核")
            head = tag + " " + ts
            body = body_text + marks
        return wrap_display(head + body, width) or [""]

    # ---- 活动行(partial) ----
    def update_active(self, text: str):
        if not self.tty:
            return
        p = self.pal
        width = self._width()
        if text:
            lag = self.metrics.lag_now
            suffix = p.grey(f"  +{lag:.1f}s") if lag > 0.05 else ""
            body = p.cyan("▶ ") + p.white(text) + suffix
        else:
            body = ""
        new_lines = wrap_display(body, width)
        self._redraw_tail(new_lines, is_active=True)

    def _redraw_tail(self, new_lines, is_active: bool):
        """重绘活动块: 先回到旧块顶部, 逐行清行重写, 光标回到内容下一行。"""
        if self.active_lines:
            self._write(f"\x1b[{len(self.active_lines)}A")
        seq = new_lines or []
        self._write("\x1b[J")  # 清掉残余
        for line in seq:
            self._write("\x1b[2K" + line + "\n")
        self.active_lines = list(seq)

    # ---- 成句固化(provisional) ----
    def commit_pending(self, sentence_id: int, text: str, start_ms: int):
        self.stats["provisional"] += 1
        lines = self._render_row(text, None, start_ms, "pending")
        if not self.tty:
            return
        # 清活动块 -> 写入待定稿行 -> 入栈
        if self.active_lines:
            self._write(f"\x1b[{len(self.active_lines)}A")
        self._write("\x1b[J")
        for line in lines:
            self._write("\x1b[2K" + line + "\n")
        self.active_lines = []
        self.stack.append((sentence_id, lines))

    # ---- 二遍定稿(final): 回刷替换对应待定稿行 ----
    def commit_final(self, sentence_id: int, text: str, speaker, start_ms, source, review):
        self.stats["final"] += 1
        if source != "qwen3-asr":
            self.stats["fallback"] += 1
        if review:
            self.stats["review"] += 1
        new_lines = self._render_row(text, speaker, start_ms, "final", source, review)

        if not self.tty:
            shown = text if text else "(空)"
            print(f"[{self._fmt_ts(start_ms)}] ({speaker}, {source}) {shown}")
            return

        # 定位目标行在栈中的物理行范围
        target_idx, up = None, len(self.active_lines)
        for i in range(len(self.stack) - 1, -1, -1):
            sid, _ = self.stack[i]
            if sid == sentence_id:
                target_idx = i
                break
            up += len(self.stack[i][1])

        if target_idx is None or up > 200:
            # 找不到对应行(异常时序/已滚出太远): 直接追加, 保证不丢内容
            if self.active_lines:
                self._write(f"\x1b[{len(self.active_lines)}A\x1b[J")
                self.active_lines = []
            for line in new_lines:
                self._write("\x1b[2K" + line + "\n")
            self.stack.append((sentence_id, new_lines))
            return

        # 回刷: 光标上移到目标行首(下方内容 + 目标旧高度), 重绘目标行至栈底+活动块
        h_old = len(self.stack[target_idx][1])
        self.stack[target_idx] = (sentence_id, new_lines)
        redraw = []
        for _, lines in self.stack[target_idx:]:
            redraw.extend(lines)
        redraw.extend(self.active_lines)
        self._write(f"\x1b[{up + h_old}A")
        for line in redraw:
            self._write("\x1b[2K" + line + "\n")

    def note(self, msg: str):
        """在活动行位置输出一条提示并保留为栈内容。"""
        if self.tty:
            if self.active_lines:
                self._write(f"\x1b[{len(self.active_lines)}A\x1b[J")
                self.active_lines = []
            for line in wrap_display(self.pal.grey(msg), self._width()):
                self._write("\x1b[2K" + line + "\n")
            self.stack.append((None, [self.pal.grey(msg)]))
        else:
            print(self.pal.grey(msg))

    def finish(self, finished_msg: dict | None, show_full: bool = False):
        p = self.pal
        if self.tty and self.active_lines:
            self._write(f"\x1b[{len(self.active_lines)}A\x1b[J")
            self.active_lines = []

        st = self.stats
        total = finished_msg.get("total_sentences") if finished_msg else st["final"]
        dur = (finished_msg or {}).get("duration_ms", 0)
        summary = [
            "会话结束" + f"  句数 {total}  时长 {dur/1000:.1f}s",
            f" partial {st['partial']}  成句 {st['provisional']}  定稿 {st['final']}"
            f"  回退 {st['fallback']}  复核 {st['review']}",
        ]
        perf = self.metrics.summarize()
        if not self.tty:
            for line in summary:
                print(line)
            for label, val in perf:
                print(f" {label:<12}: {val}")
            return
        lines = [p.bold("─" * 46), p.bold(" " + summary[0])]
        stats_line = (
            f" partial {p.cyan(str(st['partial']))}  成句 {p.cyan(str(st['provisional']))}"
            f"  定稿 {p.green(str(st['final']))}"
            f"  回退 {p.yellow(str(st['fallback']))}  复核 {p.grey(str(st['review']))}"
        )
        lines.append(stats_line)
        lines.append(p.bold("─" * 46))
        for line in lines:
            self._write("\x1b[2K" + line + "\n")

        if perf:
            self._write("\n" + p.bold("性能指标") + p.grey(" (客户端测得; 测 RTF 上限请用 ≥2 倍速推流)") + "\n")
            for label, val in perf:
                self._write(f"\x1b[2K  {p.cyan(label):<14}{val}\n")

        if show_full and finished_msg and finished_msg.get("sentences"):
            self._write("\n" + p.bold("完整转写") + p.grey(" (final 为准, 未定稿句以首遍文本显示)") + "\n")
            for s in finished_msg["sentences"]:
                spk = p.magenta(str(s.get("speaker") or "SPK?"))
                ts = p.grey(self._fmt_ts(s.get("start_ms", 0)))
                txt = s.get("text", "")
                self._write(f"\x1b[2K  {spk} {ts} {txt}\n")


# ---------------------------------------------------------------- 音频与主流程


def load_pcm16_16k(path: str) -> bytes:
    if path.lower().endswith(".wav"):
        import soundfile as sf
        import numpy as np

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if sr != TARGET_SR:
            import soxr

            mono = soxr.resample(mono, sr, TARGET_SR)
        return (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(TARGET_SR),
         "-sample_fmt", "s16", "-f", "s16le", "pipe:1"],
        capture_output=True, check=True).stdout


async def run(args) -> int:
    pcm = load_pcm16_16k(args.audio)
    if args.max_sec and args.max_sec > 0:
        pcm = pcm[: int(TARGET_SR * 2 * args.max_sec)]
    duration = len(pcm) / 2 / TARGET_SR
    chunk = int(TARGET_SR * 2 * args.chunk_ms / 1000)

    color = (not args.no_color) and sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    metrics = StreamMetrics()
    metrics.mark_audio(len(pcm) / 2 / TARGET_SR)
    view = TranscriptView(sys.stdout, color, metrics)
    view.note(
        f"连接 {args.uri} · 音频 {args.audio} ({duration:.1f}s) · 倍速 {args.rate}x"
        + (f" · 热词 {','.join(args.hotwords)}" if args.hotwords else "")
    )

    finished = asyncio.Event()

    async def receiver(ws):
        try:
            async for raw in ws:
                msg = json.loads(raw)
                t = msg.get("type")
                if t == "partial":
                    view.stats["partial"] += 1
                    metrics.on_partial(msg.get("end_ms", 0))
                    view.update_active(msg.get("text", ""))
                elif t == "provisional":
                    metrics.on_provisional(msg.get("sentence_id", 0))
                    view.commit_pending(
                        msg.get("sentence_id", 0), msg.get("text", ""), msg.get("start_ms", 0)
                    )
                elif t == "final":
                    metrics.on_final(msg.get("sentence_id", 0))
                    view.commit_final(
                        msg.get("sentence_id", 0),
                        msg.get("text", ""),
                        msg.get("speaker"),
                        msg.get("start_ms", 0),
                        msg.get("final_source", "qwen3-asr"),
                        bool(msg.get("needs_review")),
                    )
                elif t == "session_resumed":
                    view.note(f"会话已恢复 (已定稿 {msg.get('total_sentences')} 句, last_end_ms={msg.get('last_end_ms')})")
                elif t == "session_finished":
                    finished.set()
                    view.finish(msg, show_full=args.full)
                    return
                elif t == "error":
                    view.note(view.pal.red(f"服务端错误: {msg}"))
        except websockets.ConnectionClosed:
            pass

    t0 = time.time()
    async with websockets.connect(args.uri, max_size=10 * 1024 * 1024) as ws:
        start_cmd = {
            "type": "start",
            "session_id": args.session_id or f"console-{int(time.time())}",
            "language": "zh",
            "enable_spk": True,
            "expected_speakers": args.expected_speakers,
        }
        if args.hotwords:
            start_cmd["hotwords"] = args.hotwords
        await ws.send(json.dumps(start_cmd))

        recv_task = asyncio.create_task(receiver(ws))
        await asyncio.sleep(0.3)

        metrics.mark_start()
        interval = args.chunk_ms / 1000.0 / args.rate
        for i in range(0, len(pcm), chunk):
            if finished.is_set():
                break
            await ws.send(pcm[i : i + chunk])
            await asyncio.sleep(interval)
        metrics.mark_stream_end()

        await ws.send(json.dumps({"type": "stop"}))
        try:
            await asyncio.wait_for(finished.wait(), timeout=args.finish_timeout)
        except asyncio.TimeoutError:
            view.note(view.pal.red(f"等待 session_finished 超时 ({args.finish_timeout}s)"))
        recv_task.cancel()

    view.note(view.pal.grey(f"总耗时 {time.time() - t0:.1f}s"))
    return 0


def main():
    parser = argparse.ArgumentParser(description="ASRFlow 实时转写终端客户端(演示)")
    parser.add_argument("--uri", default="ws://127.0.0.1:10095")
    parser.add_argument("--audio", default="fixtures/sample_16k.wav", help="mp3/wav/flac 音频文件")
    parser.add_argument("--rate", type=float, default=1.0, help="推流倍速 (1.0=实时)")
    parser.add_argument("--max-sec", type=float, default=0.0, help="只取前 N 秒 (0=完整)")
    parser.add_argument("--chunk-ms", type=int, default=60)
    parser.add_argument("--hotwords", default="", help="逗号分隔热词")
    parser.add_argument("--expected_speakers", type=int, default=2)
    parser.add_argument("--session_id", default=None)
    parser.add_argument("--finish-timeout", type=float, default=180.0)
    parser.add_argument("--no-color", action="store_true", help="禁用彩色/光标控制")
    parser.add_argument("--full", action="store_true", help="结束时打印完整转写")
    args = parser.parse_args()

    if args.hotwords:
        args.hotwords = [w.strip() for w in args.hotwords.split(",") if w.strip()]
    else:
        args.hotwords = []

    if not os.path.exists(args.audio):
        print(f"[ERROR] audio not found: {args.audio}", file=sys.stderr)
        sys.exit(2)
    try:
        sys.exit(asyncio.run(run(args)))
    except KeyboardInterrupt:
        print("\n中断退出", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
