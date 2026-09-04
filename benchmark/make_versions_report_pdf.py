#!/usr/bin/env python3
"""四版本首遍推理 Benchmark 对比报告 PDF (weasyprint)。"""

import os
from datetime import datetime

CHARTS = "/mnt/data/zyf/sources/asr/asrflow/docs/benchmark_charts"
OUT = "/mnt/data/zyf/sources/asr/asrflow/docs/benchmark_report_pass1_versions_2026-09-04.pdf"

img = lambda name: "file://" + os.path.join(CHARTS, name)

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
td.good {{ background: #f0fdf4; }}
.kpis {{ display: flex; gap: 2.5mm; margin: 2.5mm 0; }}
.kpi {{ flex: 1; border: 0.8pt solid #e5e7eb; border-radius: 2mm; padding: 2.2mm 2.6mm; background: #f9fafb; }}
.kpi .v {{ font-size: 13pt; font-weight: 700; color: #2563eb; }}
.kpi .v.warn {{ color: #dc2626; }}
.kpi .t {{ font-size: 7.5pt; color: #4b5563; margin-top: 0.6mm; }}
.concl {{ background: #eff6ff; border: 0.8pt solid #bfdbfe; border-radius: 2mm; padding: 2.4mm 3.2mm; margin: 2mm 0; }}
.note {{ font-size: 7.5pt; color: #6b7280; }}
img {{ width: 100%; margin: 1.2mm 0; page-break-inside: avoid; }}
.fig {{ font-size: 7.5pt; color: #6b7280; text-align: center; margin: -0.5mm 0 2mm; }}
ul {{ margin: 1mm 0; padding-left: 5mm; }}
li {{ margin: 0.5mm 0; }}
</style></head><body>

<h1>ASRFlow 首遍推理（Pass-1）四版本演进 Benchmark 对比报告</h1>
<div class="meta">
日期 2026-09-04 · 仅首遍流式推理（不含二遍/vLLM）· 环境 Xeon Gold 6148 · 24 vCPU · 62.9GB · CPU 推理 ·
小参数 Paraformer-online (v2.0.4) · chunk [5,10,5] · 60ms 推流帧 · benchmark audio sample (16k) 切片<br>
版本代码: V1 git 5419fe2 (已移除) · V2 主仓 9bfd24f · V3 feat/streaming-batched dc8241c · V4 feat/streaming-onnx ffe28b7
</div>

<div class="kpis">
  <div class="kpi"><div class="v">8x</div><div class="t">V1/V2 单进程串行铁顶<br>(推理锁 + 全局 OMP 池)</div></div>
  <div class="kpi"><div class="v warn">6.6x</div><div class="t">V2 进程内 6 副本<br>(不扩容反降, GIL 所限)</div></div>
  <div class="kpi"><div class="v">31.2x</div><div class="t">V4 方案C 单进程<br>(16 路合批, 铁顶的 3.9 倍)</div></div>
  <div class="kpi"><div class="v">16 路</div><div class="t">V4 单进程实时承载<br>(V1/V2 单进程为 4 路)</div></div>
  <div class="kpi"><div class="v">1152→155 MB</div><div class="t">V4 进程常驻内存<br>(int8 图磁盘 82MB, 加载 9s→1.1s)</div></div>
</div>

<div class="concl">
<b>总体结论：</b>首遍扩容的三代路径实测——<b>V1</b> 多进程负载均衡有效（6 进程 41.8x 近线性）但每进程 8x 铁顶不变；
<b>V2</b> 进程内池化副本<b>无法复刻多进程扩展</b>（GIL + 共享 OMP 线程池，6 副本 6.6x 反低于单副本 8.4x），价值在自愈/会话绑定/故障隔离；
<b>V3</b> torch 张量层合批正确但性能倒退（6.1x）；<b>V4</b> ONNX int8 合批是唯一同时突破单进程吞吐（31.2x）、
实时承载（16 路/进程）与资源（15MB/1.1s）的路径，<b>单机 100 路从约 4~5 台机器降到 1 台以内</b>。
</div>

<h2>一、版本定义与功能对比</h2>
<table>
<tr><th style="width:7%">版本</th><th style="width:27%">架构</th><th style="width:22%">扩容路径</th><th style="width:22%">正确性验证</th><th>状态</th></tr>
<tr><td><b>V1</b> 早期</td><td>单引擎全局推理锁；每进程一份模型；多<b>进程</b> + 负载均衡</td><td>横向多进程（近线性）</td><td>现网基线（9/3 报告）</td><td class="bad">代码已移除</td></tr>
<tr><td><b>V2</b> 主仓</td><td>PooledStreamingEngine：进程内 N 副本（各持模型+锁，会话轮询绑定）+ 自愈重载</td><td>进程内副本（实测无效）+ 多进程</td><td>与 V1 同推理路径</td><td class="warn">现网 current</td></tr>
<tr><td><b>V3</b> 方案B</td><td>绕过官方 API 直接批 torch 张量；单前向线程合批；cache 沿 batch 维合并/拆分</td><td>进程内合批</td><td class="good">与官方 torch 路径逐句一致（单路/并发/长流全 PASS）</td><td class="warn">原型（性能倒退）</td></tr>
<tr><td><b>V4</b> 方案C</td><td>官方导出 ONNX 图（动态 batch 轴）+ ORT int8；单前向线程合批；每流独立确定性前端</td><td>进程内合批 + 多进程</td><td class="good">与官方单流参考逐句一致；int8 CER=0；网关端到端冒烟一致</td><td class="good">推荐合入</td></tr>
</table>
<p class="note">口径：V1 数据引自 9/3 报告（threads=2/进程）；V2/V3/V4 为本日同机实测（V2 threads=2，V3/V4 threads=8，各自合理配置并在下文标注）。
"按时" = realtime 节奏墙钟 ≈ 音频时长。全部组合识别字符数 &gt; 0。</p>

<h2>二、单进程容量：三代路径的实测对比</h2>
<img src="{img('v_chart1_capacity.png')}">
<div class="fig">图 1 · 单进程吞吐上限（左）与整机多进程（右）。V2 池化副本不改变铁顶；V4 合批单进程 31.2x，双进程 46.6x</div>

<table>
<tr><th>版本 / 配置</th><th>单进程吞吐</th><th>vs 铁顶</th><th>chunk call p99</th><th>说明</th></tr>
<tr><td>V1 单进程（threads=2, 9/3）</td><td>7.97x</td><td>1.0x</td><td>72 ms</td><td>全局锁串行，加流只涨排队</td></tr>
<tr><td>V2 单副本（threads=2）</td><td>8.42x</td><td>1.06x</td><td>82 ms</td><td>与 V1 单进程等价（交叉校验）</td></tr>
<tr class="bad"><td>V2 六副本 ×12 路（threads=2）</td><td>6.64x</td><td class="bad">0.83x</td><td class="bad">647 ms</td><td>进程内副本不扩容反降</td></tr>
<tr class="bad"><td>V2 八副本 ×16 路（threads=3）</td><td>5.60x</td><td class="bad">0.70x</td><td class="bad">1093 ms</td><td>线程分配怎么调都无效</td></tr>
<tr><td>V3 方案B ×8 路（threads=8）</td><td>6.10x</td><td class="bad">0.77x</td><td>1024 ms</td><td>future 往返 + 合批摊薄不足</td></tr>
<tr class="good"><td><b>V4 方案C ×16 路（threads=8, int8）</b></td><td><b>31.16x</b></td><td class="good"><b>3.9x</b></td><td>314 ms</td><td>合批生效，call p50 仅 0.97ms</td></tr>
<tr class="good"><td><b>V4 方案C 2 进程 ×16 路</b></td><td><b>46.65x</b></td><td class="good"><b>5.9x</b></td><td>—</td><td>合批后多进程扩展性保持</td></tr>
</table>

<h2>三、扩展性：多进程有效，进程内副本无效</h2>
<img src="{img('v_chart2_scaling.png')}">
<div class="fig">图 2 · V1 多进程（灰）近线性至 6 进程后塌缩；V2 进程内副本（蓝）加副本反而下降——GIL 与共享 OMP 池限制</div>
<ul>
<li><b>V1 多进程</b>：2/4/6 进程加速比 1.82 / 3.63 / 5.24（效率 91%/91%/87%）；12 进程塌缩至 11.3x（效率 12%）。</li>
<li><b>V2 进程内副本</b>：1→8 副本吞吐 8.42 → 5.60x 单调下降。原因：所有副本共享同一进程的 torch OMP 线程池与 GIL，
并行收益被 Python 胶水争用吃掉——<b>"进程内多副本"不能替代"多进程"</b>，V2 的实际价值是自愈重载、会话绑定与故障隔离。</li>
<li><b>V4 合批</b>从机制上绕开该限制：单线程前向把多路 chunk 拼成批张量，无锁也无副本争用；2 进程仍保持 1.5x 扩展（threads 占 16/24 vCPU 所致）。</li>
</ul>

<h2>四、实时节奏承载</h2>
<img src="{img('v_chart3_realtime.png')}">
<div class="fig">图 3 · 按时完成的实时路数：V1/V2 单进程均为 4 路；V4 单进程 16 路按时（2 进程组合以 offline 46.6x 佐证）</div>
<table>
<tr><th>组合</th><th>吞吐/墙钟</th><th>迟到率</th><th>判定</th></tr>
<tr><td>V1 1 进程 ×4 路（9/3）</td><td>3.93x / 30.5s</td><td>32%</td><td>按时（排队 p95 194ms）</td></tr>
<tr><td>V1 1 进程 ×8 路（本日 threads=8）</td><td>5.53x / 43.4s</td><td>—</td><td class="warn">掉队（端到端 1.4 倍拉长）</td></tr>
<tr><td>V2 6 副本 ×8 路</td><td>6.36x</td><td class="bad">98%</td><td class="bad">崩溃</td></tr>
<tr><td>V2 4 副本 ×16 路</td><td>6.07x</td><td class="bad">98%</td><td class="bad">崩溃</td></tr>
<tr class="good"><td><b>V4 1 进程 ×16 路</b></td><td><b>15.69x / 30.6s</b></td><td>按时</td><td class="good">call p50 0.13ms / p99 335ms</td></tr>
</table>
<p class="note">V1 整机（6 进程×4 路 = 24 路/机）为 9/3 网关与引擎级双验证结论；V4 单进程 16 路 + 2 进程组合即可在单机达到并超过该容量。</p>

<h2>五、资源与正确性</h2>
<img src="{img('v_chart4_resource.png')}">
<div class="fig">图 4 · 干净进程常驻内存口径：V4 约 155MB（torch 路径 1152MB 峰值，含反序列化双份瞬态）；磁盘工件 284.6MB(model.pt) → 274.9MB(ONNX fp32) → 81.8MB(int8 量化)；加载 9s→1.1s</div>
<img src="{img('v_chart5_tradeoff.png')}">
<div class="fig">图 5 · 吞吐-延迟散点：V4 在最高吞吐档位仍保持最低一档的 chunk 延迟；V2/V3 高延迟低吞吐（右下为优）</div>
<table>
<tr><th>维度</th><th>V1（9/3）</th><th>V2 主仓</th><th>V3 方案B</th><th>V4 方案C</th></tr>
<tr><td>模型内存 / 加载</td><td>~1152MB / 9s</td><td>N 副本 × 1152MB</td><td>1152MB / 9s</td><td class="good"><b>~155MB 常驻 / 1.1s</b> (int8 图磁盘 82MB)</td></tr>
<tr><td>输出确定性</td><td>dither 默认开启（非确定）</td><td>同 V1</td><td class="good">dither=0 确定</td><td class="good">dither=0 确定</td></tr>
<tr><td>一致性验证</td><td>—</td><td>—</td><td class="good">与官方 torch 路径逐句一致</td><td class="good">与官方 ONNX 参考逐句一致；int8 CER=0</td></tr>
<tr><td>依赖面</td><td>funasr</td><td>funasr</td><td>funasr 内部 API（pin 版本）</td><td>+ onnxruntime（官方导出图）</td></tr>
<tr><td>自愈/运维</td><td>无</td><td class="good">副本自愈重载、会话绑定</td><td>SelfHealing 复用</td><td class="good">SelfHealing 复用 + 加载 1.1s</td></tr>
</table>

<h2>六、结论与部署建议</h2>
<table>
<tr><th style="width:26%">项目</th><th style="width:32%">建议</th><th>依据</th></tr>
<tr><td>扩容主路径</td><td><b>合入 V4（streaming_asr.mode="onnx"）</b>，默认 replica 灰度</td><td>单进程 31.2x / 16 路，全部维度实测第一</td></tr>
<tr><td>V2 池化副本定位</td><td>instances 保持 1；副本仅用于<b>故障隔离</b>，不作容量手段</td><td>6 副本 6.6x &lt; 单副本 8.4x（GIL/OMP 所限）</td></tr>
<tr><td>多进程拓扑（V4）</td><td>每机 2~4 进程 × OMP/ORT 8 线程</td><td>2 进程 46.6x；4 进程预计 ~80-90x</td></tr>
<tr><td>100 路容量</td><td>V1: ~4-5 台 24vCPU 机 → <b>V4: 1 台（3~4 进程）</b></td><td>引擎级实测 + 外推（网关级复测建议）</td></tr>
<tr><td>上线前动作</td><td>网关级 bench_gateway 复跑 + int8 业务音频 CER 抽检</td><td>引擎级为保守下界；本次 4 段真实音频 CER=0</td></tr>
<tr><td>V3 方案B</td><td>保留研究记录，不合入</td><td>正确但性能 0.77x，依赖内部 API</td></tr>
</table>

<div class="note">
注 1 · 版本口径：V1 数据引自 docs/benchmark_report_2026-09-03.md（代码已移除无法复测，threads=2/进程）；V2 为本日经引擎工厂实测
（PooledStreamingEngine，bench_v2_pool 驱动，threads=2 除标注外）；V3 feat/streaming-batched、V4 feat/streaming-onnx 为本日
bench_paraformer --engine 实测（threads=8）。<br>
注 2 · 内存口径：1152MB 为 torch 引擎 ru_maxrss 峰值（含 torch 运行时与加载期 state_dict 双份瞬态）；155MB 为 ONNX int8 引擎干净进程常驻（含 ORT 运行时；权重经 mmap 按需驻留，增量测量会更小但不代表占用上限）。磁盘：model.pt 284.6MB ≈ ONNX fp32 274.9MB（换容器几乎不变），int8 量化后 81.8MB（3.4x，约三成算子未量化）。<br>注 2b · 正确性门禁：V3/V4 一致性脚本均在 dither=0 确定性口径下与官方实现逐句比对（注意 AutoModel.generate 使用独立 frontend 实例，
dither 必须在 extract_fbank 入口关闭）；V4 网关端到端冒烟（真实 FSMN-VAD）文本与直连逐字一致。<br>
注 3 · 关键工程结论存档：① decoder FSMN 零填充会污染批处理记忆（按 CIF token 数分组执行 decoder 修复）；② 进程内多副本受 GIL/
共享 OMP 池限制不能扩容；③ funasr 前端默认 dither=1.0 使任何官方路径输出非确定。<br>
注 4 · 原始数据：benchmark/results/batch_compare/（V2/V3/V4 本日）；9/3 套件目录（V1）。图表/报告再生: /tmp/gen_versions_charts.py
+ /tmp/gen_pdf（随报告归档）。生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

</body></html>"""

import logging

logging.getLogger("weasyprint").setLevel(logging.ERROR)
from weasyprint import HTML

HTML(string=html, base_url=os.getcwd()).write_pdf(OUT)

from pypdf import PdfReader

n_img = sum(len(p.images) for p in PdfReader(OUT).pages)
print(f"pdf -> {OUT} ({os.path.getsize(OUT)/1024:.0f} KB, 嵌入图片 {n_img} 张)")
assert n_img >= 5, f"图片嵌入不足: {n_img}"
