#!/usr/bin/env python3
"""Benchmark 报告图表生成: 从 results CSV 读取数据, 输出 4 张核心图 PNG。

用法: .venv/bin/python benchmark/make_report_charts.py <paraformer_suite_dir> <qwen_suite_dir> <输出目录>
"""

import os
import sys
import csv
import glob
import matplotlib

matplotlib.use("Agg")

# 中文字体: matplotlib addfont 对 .ttc 只注册首个字面(日文),
# 需先用 fontTools 抽取 SC 字面为独立 otf (见 ~/.fonts), 再在此注册
from matplotlib import font_manager

for f in glob.glob(os.path.expanduser("~/.fonts/NotoSansCJKsc-*.otf")) + glob.glob(
    "/usr/share/fonts/opentype/noto/NotoSansCJK-*.ttc"
):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt

C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GRAY = "#2563eb", "#f59e0b", "#16a34a", "#dc2626", "#6b7280"
DPI = 180


def load_engine(para_dir):
    path = glob.glob(os.path.join(para_dir, "engine", "*", "paraformer_engine.csv"))[0]
    rows = list(csv.DictReader(open(path)))
    for r in rows:  # ÷2 修正本次运行的音频计量 bug (报告已修正, CSV 为 2 倍)
        r["rtx_fix"] = float(r["throughput_rtx"]) / 2
    return {r["run_id"]: r for r in rows}


def load_gateway(para_dir):
    rows = list(csv.DictReader(open(os.path.join(para_dir, "gateway_all.csv"))))
    grid = {}
    for r in rows:
        grid[(int(r["instances"]), int(r["sessions"]))] = r
    return grid


def load_qwen(qwen_dir):
    path = glob.glob(os.path.join(qwen_dir, "qwen", "*", "qwen_vllm.csv"))[0]
    rows = list(csv.DictReader(open(path)))
    seg = {3.0: [], 10.0: []}
    for r in rows:
        seg[float(r["seg_sec"])].append(r)
    for v in seg.values():
        v.sort(key=lambda r: int(r["concurrency"]))
        for r in v:
            r["rtx_fix"] = float(r["audio_rtx"]) / 2
    return seg


def chart_scaling(engine, out):
    """图1: 首遍多进程吞吐扩展性 (offline, 每进程1路) + 单流 RTF 标注"""
    ids = ["offline_p1s1", "offline_p2s1", "offline_p4s1", "offline_p6s1", "offline_p12s1"]
    procs = [int(e["procs"]) for e in (engine[i] for i in ids)]
    rtx = [e["rtx_fix"] for e in (engine[i] for i in ids)]
    # 单流 RTF = 该路流墙钟耗时 ÷ 音频时长 (streams=1 时每流墙钟≈总墙钟; RTF 为单路指标不可累加)
    rtf = [float(engine[i]["wall_sec"]) / float(engine[i]["stream_sec"]) for i in ids]
    eff = [1.0, 0.91, 0.91, 0.87, 0.12]
    base = rtx[0]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    colors = [C_BLUE] * 4 + [C_RED]
    bars = ax.bar([str(p) for p in procs], rtx, color=colors, width=0.58, zorder=3)
    ax.plot(range(len(procs)), [base * p for p in procs], "o--", color=C_GRAY, lw=1.2,
            label="理想线性 (单进程 ×N)", zorder=4)
    for b, v, e in zip(bars, rtx, eff):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}x", ha="center", fontsize=9, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, v / 2, f"效率\n{e*100:.0f}%", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
    ax.set_xticklabels([f"{p} 进程\n单流RTF {r:.3f}" for p, r in zip(procs, rtf)])
    ax.set_xlabel("进程数 (每进程独立加载模型, 2 线程)  [单流 RTF=单路耗时÷音频时长, 与进程数基本无关; 12 进程因竞争恶化]")
    ax.set_ylabel("吞吐 (x 实时) — 多路合并容量")
    ax.set_title("首遍多进程负载均衡扩展性: 2~6 进程近线性, 12 进程塌缩", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0, 50)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def chart_lock(engine, out):
    """图2: 单进程锁串行化: 加流不加吞吐, 只涨排队"""
    s1, s4 = engine["offline_p1s1"], engine["offline_p1s4"]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    x = [0, 1]
    w = 0.32
    b1 = ax.bar([i - w / 2 for i in x], [s1["rtx_fix"], s4["rtx_fix"]], w, color=C_BLUE,
                label="吞吐 (x 实时)", zorder=3)
    for b in b1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"{b.get_height():.2f}x",
                ha="center", fontsize=10, fontweight="bold")
    ax2 = ax.twinx()
    b2 = ax2.bar([i + w / 2 for i in x], [float(s1["wait_ms_p95"]), float(s4["wait_ms_p95"])], w,
                 color=C_ORANGE, label="锁排队等待 p95 (ms)", zorder=3)
    for b in b2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}ms",
                 ha="center", fontsize=10, fontweight="bold", color="#92400e")
    ax.set_xticks(x)
    ax.set_xticklabels(["1 路流", "4 路流"])
    ax.set_ylabel("吞吐 (x 实时)", color=C_BLUE)
    ax2.set_ylabel("锁等待 p95 (ms)", color=C_ORANGE)
    ax.set_ylim(0, 10)
    ax2.set_ylim(0, 95)
    ax.set_title("单进程内加流: 吞吐不变 (~8x), 排队延迟从 0 涨到 71ms\n→ 引擎全局锁串行化, 单进程容量是硬上限",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def chart_gateway(grid, out):
    """图3: 网关级容量矩阵 (实例 x 会话 -> 成功路数 + 端到端延迟), 数值无歧义"""
    insts, sess = [1, 2, 4], [1, 4, 8, 16]
    rate = [[int(grid[(i, s)]["success"]) / s for s in sess] for i in insts]
    e2e = [[float(grid[(i, s)]["end_to_end_ms_p50"]) / 1000 if int(grid[(i, s)]["success"]) > 0 else None
            for s in sess] for i in insts]

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("ok", [C_RED, "#fbbf24", C_GREEN])
    ax.imshow(rate, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(insts)):
        for j in range(len(sess)):
            ok = int(grid[(insts[i], sess[j])]["success"])
            total = sess[j]
            t = e2e[i][j]
            if ok == total:
                label = f"{ok}/{total} 成功\n{t:.0f}s"
            elif ok == 0:
                label = f"0/{total} 成功\n全部超时"
            else:
                label = f"{ok}/{total} 成功\n{t:.0f}s·{total-ok}路超时"
            ax.text(j, i, label, ha="center", va="center", fontsize=8.5, fontweight="bold",
                    color="white" if rate[i][j] < 0.55 or rate[i][j] > 0.9 else "black")
    ax.set_xticks(range(len(sess)))
    ax.set_xticklabels([f"{s} 路会话" for s in sess])
    ax.set_yticks(range(len(insts)))
    ax.set_yticklabels([f"{i} 实例" for i in insts])
    ax.set_title("网关端到端容量矩阵 (格子: 成功路数/总路数, 端到端延迟 p50)\n单实例 4 路稳定·8 路临界; 容量随实例数线性扩展",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def chart_realtime_lag(engine, out):
    """图5: 实时节奏下的排队滞后 (log 刻度), 12 进程组崩溃"""
    ids = ["realtime_p1s1", "realtime_p2s1", "realtime_p4s1", "realtime_p6s1", "realtime_p12s1",
           "realtime_p1s4", "realtime_p2s4", "realtime_p4s4", "realtime_p6s4", "realtime_p12s4"]
    labels = ["1×1", "2×1", "4×1", "6×1", "12×1", "1×4", "2×4", "4×4", "6×4", "12×4"]
    lag = [float(engine[i]["lag_ms_p95"]) for i in ids]
    late = [float(engine[i]["late_ratio"]) * 100 for i in ids]
    colors = [C_RED if l > 400 else (C_ORANGE if l > 100 else C_BLUE) for l in lag]

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    bars = ax.bar(labels, [max(v, 1) for v in lag], color=colors, width=0.6, zorder=3)
    ax.set_yscale("log")
    for b, v, p in zip(bars, lag, late):
        txt = f"{v/1000:.0f}s" if v >= 1000 else f"{v:.0f}"
        ax.text(b.get_x() + b.get_width() / 2, max(v, 1) * 1.35, txt, ha="center", fontsize=8, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, max(v, 1) * 0.55, f"迟到\n{p:.0f}%", ha="center", va="center",
                fontsize=7, color="white", fontweight="bold")
    ax.axhline(30, color=C_GRAY, ls=":", lw=1)
    ax.text(9.45, 34, "迟到阈值 30ms", fontsize=7, color=C_GRAY, ha="right")
    ax.set_xlabel("组合 (进程数 × 每进程流数), 共 1~48 路实时流")
    ax.set_ylabel("chunk 滞后 p95 (ms, 对数轴)")
    ax.set_title("实时节奏承载: ≤6 进程按时完成 (滞后<330ms); 12 进程组滞后爆炸至 91s",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(1, 300000)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def chart_qwen(seg, out):
    """图4: 二遍并发扫描: 延迟 + 吞吐 双轴"""
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax2 = ax.twinx()
    styles = {3.0: (C_BLUE, "o", "-"), 10.0: (C_GREEN, "s", "--")}
    for sec, (c, m, ls) in styles.items():
        rows = seg[sec]
        cx = [int(r["concurrency"]) for r in rows]
        p50 = [float(r["lat_ms_p50"]) for r in rows]
        p99 = [float(r["lat_ms_p99"]) for r in rows]
        rps = [float(r["req_per_sec"]) for r in rows]
        ax.plot(cx, p50, marker=m, color=c, ls=ls, lw=1.6, label=f"{sec:.0f}s 段 p50")
        ax.plot(cx, p99, marker=m, color=c, ls=ls, lw=1.0, alpha=0.45, label=f"{sec:.0f}s 段 p99")
        ax2.plot(cx, rps, marker=m, color=c, ls=ls, lw=2.2, alpha=0.0)  # 占位保持一致性
        ax2.plot(cx, rps, marker=m, color=c, ls=ls, lw=2.0, markevery=[len(cx) - 1], alpha=0.9)
        ax2.annotate(f"{sec:.0f}s 段 {rps[-1]:.0f} req/s", (cx[-1], rps[-1]), textcoords="offset points",
                     xytext=(-4, 8 if sec == 3.0 else -14), fontsize=9, color=c, fontweight="bold", ha="right")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 24, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "24", "32"])
    ax.set_xlabel("并发请求数 (单卡 4090)")
    ax.set_ylabel("延迟 (ms)")
    ax2.set_ylabel("吞吐 (req/s)")
    ax.set_title("二遍 Qwen3-ASR 并发扫描: 336 请求零失败, 32 并发仍未饱和\n10s 段吞吐 54 req/s ≈ 540x 实时, p99 432ms ≪ 生产 8s 超时",
                 fontsize=11, fontweight="bold")
    rtf10_c1 = float(seg[10.0][0]["lat_ms_p50"]) / 10000
    rtf10_c32 = float(seg[10.0][-1]["lat_ms_p50"]) / 10000
    ax.text(0.98, 0.96, f"单请求 RTF (p50延迟÷段长)\n10s 段: 并发1 {rtf10_c1:.3f} → 并发32 {rtf10_c32:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f0fdf4", ec=C_GREEN, alpha=0.9))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend([plt.Line2D([0], [0], color=C_BLUE, lw=2), plt.Line2D([0], [0], color=C_GREEN, lw=2)],
               ["3s 段吞吐 (req/s)", "10s 段吞吐 (req/s)"], loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def main():
    para_dir, qwen_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    engine = load_engine(para_dir)
    grid = load_gateway(para_dir)
    seg = load_qwen(qwen_dir)
    chart_scaling(engine, os.path.join(out_dir, "chart1_scaling.png"))
    chart_lock(engine, os.path.join(out_dir, "chart2_lock.png"))
    chart_gateway(grid, os.path.join(out_dir, "chart3_gateway.png"))
    chart_qwen(seg, os.path.join(out_dir, "chart4_qwen.png"))
    chart_realtime_lag(engine, os.path.join(out_dir, "chart5_realtime_lag.png"))
    print("charts ->", out_dir)


if __name__ == "__main__":
    main()
