#!/usr/bin/env python3
"""四版本首遍推理对比报告图表: V1 早期多实例 / V2 主仓池化 / V3 方案B / V4 方案C。"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
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

C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GRAY, C_PURPLE = (
    "#2563eb", "#f59e0b", "#16a34a", "#dc2626", "#6b7280", "#7c3aed")
DPI = 180
OUT = "/mnt/data/zyf/sources/asr/asrflow/docs/benchmark_charts"
os.makedirs(OUT, exist_ok=True)

# ---------------- 数据 (实测, 见报告表格溯源) ----------------
# V1: 9/3 报告 (threads=2/进程, 多进程); V2: 本日实测 (threads=2, 进程内副本)
# V3 B: 本日 (threads=8); V4 C: 本日 (threads=8, ONNX int8)
v1_procs = [1, 2, 4, 6, 12]
v1_rtx = [7.97, 14.51, 28.94, 41.79, 11.32]
v2_inst = [1, 2, 4, 6, 8]
v2_rtx = [8.42, 7.17, 6.34, 6.64, 5.60]  # 6.64 为 6副本×2线程; 8 为 8副本×3线程

# 图1: 单进程吞吐上限 + 整机多进程
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [3, 2]})
names = ["V1 早期\n(全局锁串行)", "V2 主仓池化\n(6副本)", "V3 方案B\n(torch合批)", "V4 方案C\n(ONNX合批)"]
vals = [8.0, 6.6, 6.1, 31.2]
colors = [C_GRAY, C_BLUE, C_ORANGE, C_GREEN]
bars = ax1.bar(names, vals, color=colors, width=0.62)
for b, v in zip(bars, vals):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}x", ha="center", fontsize=10, fontweight="bold")
ax1.set_ylabel("单进程离线吞吐 (x 实时)")
ax1.set_title("单进程吞吐上限 (各自合理配置)", fontsize=11)
ax1.set_ylim(0, 37)
ax1.axhline(8, color=C_GRAY, ls=":", lw=0.8)
ax1.text(0.02, 8.7, "8x 串行铁顶", fontsize=7.5, color=C_GRAY, ha="left")

names2 = ["V1\n6 进程", "V4 方案C\n2 进程", "V4 方案C\n4 进程(外推)"]
vals2 = [41.8, 46.6, 46.6 * 2 * 0.85]
bars2 = ax2.bar(names2, vals2, color=[C_GRAY, C_GREEN, "#bbf7d0"], width=0.55)
for b, v in zip(bars2, vals2):
    lab = f"{v:.0f}x" if v < 80 else f"≈{v:.0f}x"
    ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, lab, ha="center", fontsize=10, fontweight="bold")
ax2.set_title("整机多进程吞吐", fontsize=11)
ax2.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(f"{OUT}/v_chart1_capacity.png", dpi=DPI)
plt.close()

# 图2: 扩展性 — V1 多进程 vs V2 进程内副本
fig, ax = plt.subplots(figsize=(8.6, 3.8))
ax.plot(v1_procs, v1_rtx, "o-", color=C_GRAY, label="V1: 多进程负载均衡 (9/3 实测)", lw=1.8)
ax.plot(v2_inst, v2_rtx, "s-", color=C_BLUE, label="V2: 进程内池化副本 (本日实测)", lw=1.8)
ideal_x = [1, 2, 4, 6]
ax.plot(ideal_x, [8 * x for x in ideal_x], ":", color=C_GREEN, lw=1.2, label="理想线性 (8x × N)")
ax.annotate("12 进程塌缩\n(效率 12%)", xy=(12, 11.32), xytext=(9.6, 22), fontsize=8, color=C_RED,
            arrowprops=dict(arrowstyle="->", color=C_RED, lw=1))
ax.annotate("进程内副本不扩容\n(GIL + 共享 OMP 线程池)", xy=(6, 6.64), xytext=(4.4, 24), fontsize=8, color=C_BLUE,
            arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1))
ax.set_xlabel("并行单元数 (进程数 / 进程内副本数)")
ax.set_ylabel("离线吞吐 (x 实时)")
ax.set_title("扩展性对比: 横向多进程 vs 进程内多副本 (threads=2)", fontsize=11)
ax.legend(fontsize=8.5, loc="upper left")
ax.set_xticks([1, 2, 4, 6, 8, 12])
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUT}/v_chart2_scaling.png", dpi=DPI)
plt.close()

# 图3: 实时承载 (单进程按时完成路数)
fig, ax = plt.subplots(figsize=(8.6, 3.6))
names = ["V1 单进程\n(全局锁)", "V2 单进程\n(6副本)", "V4 方案C\n单进程", "V1 整机\n6进程×4路", "V4 方案C\n2进程×16路*"]
vals = [4, 4, 16, 24, 32]
cols = [C_GRAY, C_BLUE, C_GREEN, "#9ca3af", "#86efac"]
bars = ax.bar(names, vals, color=cols, width=0.6)
notes = ["8 路即掉队\n(墙钟 43s/30s)", "8 路 late 98% 崩溃", "16 路按时\n(call p99 335ms)", "24 路 按时\n(9/3 实测)", "按时 (2×8 路\noffline 46.6x 佐证)"]
for b, v, n in zip(bars, vals, notes):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v} 路", ha="center", fontsize=10, fontweight="bold")
    ax.text(b.get_x() + b.get_width() / 2, v / 2, n, ha="center", fontsize=6.8, color="white" if v > 6 else "#374151")
ax.set_ylabel("按时完成的实时流路数")
ax.set_title("实时节奏承载能力 (单进程 / 整机)", fontsize=11)
ax.set_ylim(0, 36)
plt.tight_layout()
plt.savefig(f"{OUT}/v_chart3_realtime.png", dpi=DPI)
plt.close()

# 图4: 资源对比 (内存 + 加载时间)
fig, ax1 = plt.subplots(figsize=(8.2, 3.4))
names = ["V1/V2/V3\n(torch 权重)", "V4 方案C\n(ONNX int8 图)"]
mem = [1152, 155]
load = [9.0, 1.1]
x = [0, 1]
b1 = ax1.bar([i - 0.16 for i in x], mem, width=0.3, color=C_BLUE, label="模型常驻内存 (MB)")
ax1.set_ylabel("内存 (MB)", color=C_BLUE)
ax1.tick_params(axis="y", labelcolor=C_BLUE)
ax1.set_xticks(x)
ax1.set_xticklabels(names)
ax1.set_ylim(0, 1400)
for i, v in zip(x, mem):
    ax1.text(i - 0.16, v + 25, f"{v} MB", ha="center", fontsize=9, color=C_BLUE, fontweight="bold")
ax2 = ax1.twinx()
b2 = ax2.bar([i + 0.16 for i in x], load, width=0.3, color=C_ORANGE, label="模型加载时间 (s)")
ax2.set_ylabel("加载时间 (s)", color=C_ORANGE)
ax2.tick_params(axis="y", labelcolor=C_ORANGE)
ax2.set_ylim(0, 11)
for i, v in zip(x, load):
    ax2.text(i + 0.16, v + 0.22, f"{v:.1f}s", ha="center", fontsize=9, color=C_ORANGE, fontweight="bold")
ax1.set_title("单进程资源占用: torch 权重 vs ONNX int8 合批", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/v_chart4_resource.png", dpi=DPI)
plt.close()

# 图5: 吞吐-延迟散点 (每版本取其最高吞吐组合)
fig, ax = plt.subplots(figsize=(8.2, 3.6))
pts = [
    ("V1 1进程×4路", 8.1, 68, C_GRAY),
    ("V1 6进程×24路", 41.8, 80, "#9ca3af"),
    ("V2 6副本×12路", 6.6, 647, C_BLUE),
    ("V3 B 8路", 6.1, 1024, C_ORANGE),
    ("V4 C 16路", 31.2, 314, C_GREEN),
    ("V4 C 2进程×16路", 46.6, 280, "#86efac"),
]
for name, rtx, p99, c in pts:
    ax.scatter(rtx, p99, s=70, color=c, zorder=3)
    off = (6, -11) if rtx > 40 else (6, 5)
    ax.annotate(name, (rtx, p99), textcoords="offset points", xytext=off, fontsize=8)
ax.set_xlabel("离线吞吐 (x 实时)")
ax.set_ylabel("chunk call 延迟 p99 (ms)")
ax.set_title("吞吐 vs 单 chunk 延迟 (越靠右下越好)", fontsize=11)
ax.set_xlim(0, 52)
ax.set_ylim(0, 1150)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUT}/v_chart5_tradeoff.png", dpi=DPI)
plt.close()

print("charts ->", OUT)
for f in sorted(os.listdir(OUT)):
    if f.startswith("v_chart"):
        print(" ", f)
