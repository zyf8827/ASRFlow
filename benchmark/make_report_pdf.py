#!/usr/bin/env python3
"""精简版 Benchmark 报告 PDF 生成: 读取 results CSV (÷2 修正) + 预生成图表, 经 weasyprint 渲染。

用法: .venv/bin/python benchmark/make_report_pdf.py <paraformer_suite_dir> <qwen_suite_dir> <charts_dir> <输出.pdf>
依赖: 图表先由 make_report_charts.py 生成; 系统需 Noto Sans CJK SC 字体。
注意: 图片用绝对 file:// 路径嵌入 (相对路径 + base_url 会静默丢图)。
"""

import os
import sys
import csv
import glob
from datetime import datetime

CHARTS = ["chart1_scaling.png", "chart2_lock.png", "chart5_realtime_lag.png",
          "chart3_gateway.png", "chart4_qwen.png"]


def load_engine(para_dir):
    path = glob.glob(os.path.join(para_dir, "engine", "*", "paraformer_engine.csv"))[0]
    out = {}
    for r in csv.DictReader(open(path)):
        r["rtx_fix"] = float(r["throughput_rtx"]) / 2  # 本次运行 CSV 的音频计量为 2 倍, 已修正
        out[r["run_id"]] = r
    return out


def load_gateway(para_dir):
    out = {}
    for r in csv.DictReader(open(os.path.join(para_dir, "gateway_all.csv"))):
        out[(int(r["instances"]), int(r["sessions"]))] = r
    return out


def load_qwen(qwen_dir):
    path = glob.glob(os.path.join(qwen_dir, "qwen", "*", "qwen_vllm.csv"))[0]
    rows = list(csv.DictReader(open(path)))
    seg = {3.0: [], 10.0: []}
    for r in rows:
        seg[float(r["seg_sec"])].append(r)
    for v in seg.values():
        v.sort(key=lambda r: int(r["concurrency"]))
    return seg


def build_html(engine, gateway, qwen, charts_dir):
    # ---- 首遍扩展表 (offline, 每进程1路) + 单流 RTF
    scale_ids = ["offline_p1s1", "offline_p2s1", "offline_p4s1", "offline_p6s1", "offline_p12s1"]
    eff = [1.00, 0.91, 0.91, 0.87, 0.12]
    scale_rows = ""
    for rid, e in zip(scale_ids, eff):
        r = engine[rid]
        rtx = r["rtx_fix"]
        rtf_stream = float(r["wall_sec"]) / float(r["stream_sec"])  # 单流 RTF (streams=1: 每流墙钟≈总墙钟)
        cls = ' class="bad"' if r["procs"] == "12" else ""
        scale_rows += (
            f"<tr{cls}><td>{r['procs']}</td><td><b>{rtx:.1f}x</b></td>"
            f"<td>{rtf_stream:.3f}</td>"
            f"<td>{float(rtx)/engine['offline_p1s1']['rtx_fix']:.2f}</td>"
            f"<td>{e*100:.0f}%</td><td>{float(r['call_ms_p99']):.0f}</td></tr>"
        )

    # ---- 网关表 (关键档, 无歧义表述) + 单会话 RTF
    gw_rows = ""
    for inst, sess in [(1, 1), (1, 4), (1, 8), (1, 16), (2, 8), (2, 16), (4, 8), (4, 16)]:
        r = gateway[(inst, sess)]
        ok, total = int(r["success"]), int(r["sessions"])
        fp = float(r["first_partial_ms_p50"]) if ok > 0 else None
        e2e = float(r["end_to_end_ms_p50"]) / 1000 if ok > 0 else None
        # 单会话 RTF = 该会话端到端耗时 ÷ 音频时长 (30s)
        sess_rtf = e2e / float(r["stream_sec"]) if e2e else None
        if ok == total:
            verdict, cls = f"✔ {ok}/{total} 全部成功", ""
        elif ok == 0:
            verdict, cls = f"✘ 0/{total} 全部超时", ' class="bad"'
        else:
            verdict, cls = f"◐ {ok}/{total} 成功, 其中 {total-ok} 路超时", ' class="warn"'
        gw_rows += (
            f"<tr{cls}><td>{inst}</td><td>{sess}</td><td>{verdict}</td>"
            f"<td>{f'{fp:.0f}' if fp else '—'}</td>"
            f"<td>{f'{e2e:.1f}' if e2e else '—'}</td>"
            f"<td>{f'{sess_rtf:.2f}' if sess_rtf else '—'}</td></tr>"
        )

    # ---- 二遍表: 10s 段为主 + 3s 段, 含 RTF
    q_rows = ""
    for r3, r10 in zip(qwen[3.0], qwen[10.0]):
        rtf10 = float(r10["lat_ms_p50"]) / (10.0 * 1000)
        rtf3 = float(r3["lat_ms_p50"]) / (3.0 * 1000)
        q_rows += (
            f"<tr><td>{r10['concurrency']}</td>"
            f"<td>{float(r10['lat_ms_p50']):.0f} / {float(r10['lat_ms_p99']):.0f}</td>"
            f"<td>{rtf10:.3f}</td>"
            f"<td><b>{float(r10['req_per_sec']):.1f}</b></td><td>{float(r10['audio_rtx'])/2:.0f}x</td>"
            f"<td>{float(r3['lat_ms_p50']):.0f} / {float(r3['lat_ms_p99']):.0f}</td>"
            f"<td>{rtf3:.3f}</td><td>{float(r3['req_per_sec']):.1f}</td></tr>"
        )

    img = lambda name: "file://" + os.path.abspath(os.path.join(charts_dir, name))
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><style>
@page {{
  size: A4; margin: 15mm 13mm 17mm 13mm;
  @bottom-center {{ content: counter(page) " / " counter(pages); font-size: 8pt; color: #888;
                    font-family: 'Noto Sans CJK SC', sans-serif; }}
}}
* {{ box-sizing: border-box; }}
body {{ font-family: 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', sans-serif; font-size: 9.5pt;
        color: #1f2937; line-height: 1.5; margin: 0; }}
h1 {{ font-size: 16.5pt; margin: 0 0 2mm; }}
h2 {{ font-size: 12pt; margin: 5.5mm 0 2mm; padding-left: 2.5mm; border-left: 3.5pt solid #2563eb;
      page-break-after: avoid; }}
h3 {{ font-size: 10pt; margin: 3mm 0 1mm; color: #374151; page-break-after: avoid; }}
.meta {{ color: #6b7280; font-size: 8pt; margin-bottom: 2.5mm; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 2mm 0; page-break-inside: avoid; }}
th {{ background: #eff6ff; color: #1e40af; font-weight: 600; }}
th, td {{ border: 0.5pt solid #d1d5db; padding: 1.1mm 1.8mm; text-align: center; }}
td.bad {{ background: #fef2f2; }}
td.warn {{ background: #fffbeb; }}
.kpis {{ display: flex; gap: 2.5mm; margin: 2.5mm 0; }}
.kpi {{ flex: 1; border: 0.8pt solid #e5e7eb; border-radius: 2mm; padding: 2.2mm 2.6mm; background: #f9fafb; }}
.kpi .v {{ font-size: 13pt; font-weight: 700; color: #2563eb; }}
.kpi .v.warn {{ color: #dc2626; }}
.kpi .t {{ font-size: 7.5pt; color: #4b5563; margin-top: 0.6mm; }}
.concl {{ background: #eff6ff; border: 0.8pt solid #bfdbfe; border-radius: 2mm; padding: 2.4mm 3.2mm; margin: 2mm 0; }}
.metrics {{ background: #f8fafc; border: 0.8pt dashed #94a3b8; border-radius: 2mm; padding: 2.2mm 3.2mm;
            margin: 2mm 0; font-size: 8pt; }}
.metrics td {{ border: none; text-align: left; padding: 0.7mm 1.5mm 0.7mm 0; }}
.metrics .f {{ font-family: 'DejaVu Sans Mono', monospace; color: #1e40af; white-space: nowrap; }}
.note {{ font-size: 7.5pt; color: #6b7280; }}
img {{ width: 100%; margin: 1.2mm 0; page-break-inside: avoid; }}
.fig {{ font-size: 7.5pt; color: #6b7280; text-align: center; margin: -0.5mm 0 2mm; }}
ul {{ margin: 1mm 0; padding-left: 5mm; }}
li {{ margin: 0.5mm 0; }}
</style></head><body>

<h1>ASRFlow 双遍模型独立性能压测报告</h1>
<div class="meta">
日期 2026-09-03 · 代码 git 5419fe2d · 配置基准 config/config.default.yaml · 音频 benchmark audio sample ·
首遍: Xeon Gold 6148 · 24 vCPU · 62.9GB · CPU 推理 | 二遍: qwen3-asr-1.7b @ 127.0.0.1:8899 · vLLM · 单卡 RTX 4090
</div>

<div class="kpis">
  <div class="kpi"><div class="v">RTF 0.125</div><div class="t">首遍单路 RTF<br>(单流推理快于实时 8 倍)</div></div>
  <div class="kpi"><div class="v">41.8x</div><div class="t">6 进程总吞吐 (容量)<br>≈ 24 路实时流, 效率 87%</div></div>
  <div class="kpi"><div class="v warn">12%</div><div class="t">12 进程并行效率<br>(吞吐塌缩, 应避免)</div></div>
  <div class="kpi"><div class="v">4 → 16 路</div><div class="t">网关 1 → 4 实例容量<br>(负载均衡线性扩展)</div></div>
  <div class="kpi"><div class="v">RTF 0.041</div><div class="t">二遍单请求 RTF<br>(32 并发, 零错误未饱和)</div></div>
</div>

<div class="concl">
<b>总体结论：</b>系统瓶颈完全在首遍 CPU 侧——单进程推理被引擎内部全局锁串行化（单路 RTF 0.125，吞吐上限 8x 实时），
扩容唯一有效路径是<b>多进程负载均衡</b>（2~6 进程近线性，本机拐点在 6~12 之间，12 进程反而塌缩）；
二遍单卡 4090 容量是首遍单实例的 <b>50 倍以上</b>（540x vs 8x），当前生产限流参数（max_concurrency=8）过于保守，
<b>二遍无需任何扩容</b>。
</div>

<h3>核心指标口径（计算方式）</h3>
<table class="metrics">
<tr><td class="f">RTF = 单路处理耗时 ÷ 该路音频时长</td><td>ASR 标准指标，&lt;1 即快于实时。<b>单路/单请求指标，不可跨路累加</b>——并发下每路 RTF 保持基本恒定或因排队升高，不会随吞吐增长而下降。本报告口径：引擎级 = 单流墙钟÷流时长；网关级 = 单会话端到端÷会话音频时长；二遍 = 单请求延迟÷段长。</td></tr>
<tr><td class="f">吞吐 (x 实时) = 音频总秒数 ÷ 墙钟秒数</td><td><b>多路合并容量指标</b>（非 RTF）：每秒可消化的音频秒数，≈ 可支撑的实时流路数；仅单路运行时才等于 1/RTF。</td></tr>
<tr><td class="f">call_ms / wait_ms</td><td>单次 process_chunk 总耗时 / 其中锁排队耗时（代理锁分离计量）。单流时 wait≈0，call 即纯推理延迟。</td></tr>
<tr><td class="f">lag_ms / 迟到%</td><td>实时节奏下 chunk 实际发出时刻晚于真实语速调度时刻的滞后；迟到 = lag &gt; 30ms（半个 chunk 周期）的占比。</td></tr>
<tr><td class="f">加速比 / 并行效率</td><td>N 进程吞吐 ÷ 1 进程吞吐；效率 = 加速比 ÷ N，≥90% 视为线性。</td></tr>
<tr><td class="f">首 partial / 端到端</td><td>首帧音频发出 → 首个非空 partial 返回 / → session_finished 收尾完成。</td></tr>
</table>

<h2>一、首遍 Paraformer (CPU) — 多进程扩展性</h2>
<img src="{img('chart1_scaling.png')}">
<div class="fig">图 1 · 多进程吞吐扩展性（含 RTF 与并行效率标注；12 进程红色=塌缩）</div>
<table>
<tr><th>进程数</th><th>吞吐 (x 实时)</th><th>单流 RTF</th><th>加速比</th><th>并行效率</th><th>chunk p99 (ms)</th></tr>
{scale_rows}
</table>
<p class="note">单流 RTF 在 1~6 进程间保持 0.125~0.144 基本恒定（并发不改变单路体验）；12 进程恶化至 1.06（每路耗时反超音频时长）。</p>

<h3>单进程饱和机制：加流不加吞吐，只涨排队（引擎全局锁）</h3>
<img src="{img('chart2_lock.png')}">
<div class="fig">图 2 · 单进程 1 路 vs 4 路：吞吐同为 ~8x（RTF ~0.125），锁排队从 0 涨到 71ms</div>

<h3>实时节奏承载（按真实语速推流）</h3>
<img src="{img('chart5_realtime_lag.png')}">
<div class="fig">图 3 · 各组合 chunk 滞后 p95（对数轴）：≤6 进程按时完成，12 进程组滞后爆炸</div>
<ul>
<li>2/4/6 进程加速比 1.82 / 3.63 / 5.24（效率 91% / 91% / 87%）——<b>多进程负载均衡可线性扩展</b>。</li>
<li>12 进程 × 2 线程恰好占满 24 vCPU，无调度余量 → 吞吐塌缩至 11.3x（低于 2 进程），chunk p99 恶化 13 倍。</li>
<li>实时节奏验证：6 进程 × 4 路 = 24 路实时流整体按时完成（墙钟 30.8s vs 音频 30s）；单进程 4 路为稳态上限。</li>
</ul>

<h2>二、首遍网关级 — 多实例负载均衡端到端验证</h2>
<img src="{img('chart3_gateway.png')}">
<div class="fig">图 4 · 网关容量矩阵（格子 = 成功路数/总路数 + 端到端延迟 p50；绿=全部成功，红=全部超时）</div>
<table>
<tr><th>实例</th><th>会话</th><th>结果（成功路数/总路数）</th><th>首 partial p50 (ms)</th><th>端到端 p50 (s)</th><th>单会话 RTF p50</th></tr>
{gw_rows}
</table>
<ul>
<li><b>容量随实例数线性</b>：单实例 4 路稳定（8 路临界）→ 2 实例 8 路 → 4 实例 16 路全部成功。</li>
<li>单会话 RTF 随负载直观恶化：1 实例 1 路 1.01 → 4 路 1.31 → 8 路 2.40（成功会话的端到端耗时已达音频时长的 2.4 倍，积压明显）；4 实例 16 路仍维持 1.39。</li>
<li>首 partial 延迟全程稳定 0.65~1.5s（VAD 480ms 判定主导），负载增加不劣化直至饱和。</li>
<li>过载模式为端到端时间无界拉长直至 90s 收尾超时（非崩溃）→ 生产应配单实例并发上限主动拒绝。</li>
<li>注：网关有效容量（~4 路/实例）约为引擎纯吞吐（8x）的一半——VAD 每帧推理、断句 flush 额外前向、协议开销叠加，且实时会话有截止时间（排队论上稳定运行需利用率 &lt;70%）。</li>
</ul>

<h2>三、二遍 Qwen3-ASR (远程 vLLM · 单卡 4090) — 并发扫描</h2>
<img src="{img('chart4_qwen.png')}">
<div class="fig">图 5 · 延迟分位 + 吞吐双轴（10s/3s 两种段长），右上角标注单请求 RTF</div>
<table>
<tr><th rowspan="2">并发</th><th colspan="3">10s 段（长句）</th><th colspan="3">3s 段（短句）</th><th rowspan="2">系统吞吐<br>(req/s, 10s)</th></tr>
<tr><th>p50/p99 (ms)</th><th>RTF p50</th><th>系统实时倍率</th><th>p50/p99 (ms)</th><th>RTF p50</th><th>—</th></tr>
{q_rows}
</table>
<ul>
<li><b>336/336 请求全部成功、零超时</b>（30s 超时口径）；并发 32 仍未出现吞吐平台期（vLLM continuous batching 持续吃满 batch）。</li>
<li>单请求 RTF：10s 段并发 1 为 0.026、并发 32 仅 0.041 —— 即使满并发仍比实时快 24 倍以上。</li>
<li>全表最大延迟 442ms，距生产 hard_timeout=8s 有 <b>18 倍余量</b>；并发 1→2 时延迟反而下降（GPU 批处理摊薄开销）。</li>
<li>容量换算：系统级 540x 实时 ≈ 单卡支撑 540 路"每 10s 一句"的二遍实时流，为网关单实例容量（4~8 路）的 50 倍以上。</li>
</ul>

<h2>四、部署建议</h2>
<table>
<tr><th style="width:22%">项目</th><th style="width:30%">建议值</th><th>依据</th></tr>
<tr><td>首遍进程拓扑</td><td><b>每机 4~6 实例</b>，每实例 OMP 线程=2</td><td>2~6 进程效率 87~91%；12 进程塌缩至 12%</td></tr>
<tr><td>单实例并发上限</td><td><b>4~6 路</b>（超出排队/拒绝）</td><td>网关实测 4 路零失败、8 路出现 1 路超时</td></tr>
<tr><td>单机系统容量</td><td><b>≈ 24 路实时流</b></td><td>引擎级 6×4=24 路按时完成</td></tr>
<tr><td>二遍限流</td><td>max_concurrency <b>8 → 16~32</b></td><td>32 并发 RTF 0.041、零错误</td></tr>
<tr><td>二遍超时</td><td>hard_timeout 8s 维持</td><td>最大延迟 442ms，余量 18 倍</td></tr>
</table>

<div class="note">
注 1 · 数据修正：本次运行脚本存在音频量计量 bug（字节除采样率未除 2），原始 CSV 中 throughput_rtx / audio_rtx / total_audio_sec 为真实值 2 倍；
本报告所有数值已按 ÷2 修正，RTF/加速比/效率等相对指标不受影响。脚本已修复（bench_paraformer.py / bench_gateway.py）。<br>
注 2 · 测试方法：引擎级绕过网关直测模型（进程独立加载模型、预热 5s 后同步起表计时，计时窗口不含模型加载与预热）；
网关级 final_asr=mock 隔离二遍、VAD 静音阈值 480ms、说话人关闭；二遍直连生产同款 Qwen3ASREngine（multipart transcriptions API）。
全部组合识别字符数 &gt; 0、零引擎错误。<br>
注 3 · 原始数据：benchmark/results/20260903_185408_paraformer_suite/ 与 20260903_192556_qwen_suite/（含逐 chunk / 逐请求明细 JSON 与 manifest 溯源）。详版数据报告: docs/benchmark_report_2026-09-03.md。<br>
注 4 · 复现：bash benchmark/run_paraformer_suite.sh ; bash benchmark/run_qwen_suite.sh；图表/报告再生:
benchmark/make_report_charts.py + make_report_pdf.py。生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

</body></html>"""
    return html


def main():
    para_dir, qwen_dir, charts_dir, out_pdf = sys.argv[1:5]
    engine = load_engine(para_dir)
    gateway = load_gateway(para_dir)
    qwen = load_qwen(qwen_dir)
    html = build_html(engine, gateway, qwen, charts_dir)

    import logging

    logging.getLogger("weasyprint").setLevel(logging.ERROR)
    from weasyprint import HTML

    doc = HTML(string=html, base_url=os.getcwd())
    doc.write_pdf(out_pdf)

    # 客观校验: 确认图片已嵌入 (旧版相对路径曾静默丢图)
    from pypdf import PdfReader

    try:
        n_img = sum(len(p.images) for p in PdfReader(out_pdf).pages)
        print(f"pdf -> {out_pdf} ({os.path.getsize(out_pdf)/1024:.0f} KB, 嵌入图片 {n_img} 张)")
        assert n_img >= len(CHARTS), f"图片嵌入数量不足: {n_img}"
    except ImportError:
        print(f"pdf -> {out_pdf} ({os.path.getsize(out_pdf)/1024:.0f} KB) [pypdf 未装, 跳过图片校验]")


if __name__ == "__main__":
    main()
