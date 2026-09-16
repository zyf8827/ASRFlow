#!/usr/bin/env python3
"""
ASRFlow 本机实例健康检查。

自动发现本机启动的全部 asrflow 实例, 逐个做两级检查 (--endpoint 直接探测指定地址,
跳过发现与 HTTP 探活):
  1) HTTP 探活: /healthz (进程活) + /ready (引擎就绪; 503 会带原因)
  2) WS 功能探测: 用一段真实音频 (默认 fixtures/sample_16k.wav 前 N 秒, 倍速推流) 走
     start -> 推流 -> commit/stop -> session_finished 完整链路, 收到非空识别
     文本即判定功能可用 (不要求二遍来源, mock/降级文本也算)。

实例发现来源 (按 ws 端口去重):
  - docker: 正在运行且 名称或镜像 含 asrflow 的容器, 端口取容器 env
    ASR_SERVER_PORT/ASR_HTTP_PORT; host 网络直接用, bridge 网络解析宿主映射,
    未映射则退回容器 IP。
  - process: /proc 扫描本仓库 main.py 进程 (按 main.py 同级存在 core/ 与
    pipeline/ 目录确认是本仓库, 避免误报其他项目同名脚本), 端口解析自
    --port/--http_port 参数。支持后台 daemon 启动与手工前台启动。

  --endpoint ws://host:port 时跳过以上发现与 HTTP 探活, 只对该地址做 WS 功能探测
  (容器内运行: docker exec asrflow python3 scripts/healthcheck.py
  --endpoint ws://127.0.0.1:10095)。

用法:
    .venv/bin/python scripts/healthcheck.py                 # 自动发现 + 全量检查
    python3 scripts/healthcheck.py --list                  # 只发现实例 + HTTP 探活
    python3 scripts/healthcheck.py --ws-ports 10095,10105  # 手动指定 (跳过自动发现)
    python3 scripts/healthcheck.py --endpoint ws://127.0.0.1:10095  # 直接探测指定地址
    python3 scripts/healthcheck.py --max-sec 15 --rate 8   # 更短的探测音频/更快推流

退出码: 0=全部实例功能可用; 1=存在不可用实例或未发现实例; 2=环境错误
        (缺音频/ffmpeg/websockets)。
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WS_PORT = 10095
DEFAULT_HTTP_PORT = 10096
TARGET_SR = 16000


# ---------------------------------------------------------------------------
# 实例发现
# ---------------------------------------------------------------------------
class Instance:
    def __init__(self, source, name, ws_host, ws_port, http_host, http_port, note=""):
        self.source = source        # docker | process | manual
        self.name = name
        self.ws_host, self.ws_port = ws_host, ws_port
        self.http_host, self.http_port = http_host, http_port
        self.note = note

    @property
    def key(self):
        return (self.ws_host, self.ws_port)

    def label(self):
        return f"{self.name} (ws {self.ws_host}:{self.ws_port})"


def find_docker_instances():
    """发现运行中的 asrflow 容器; docker 不可用/无权限时静默跳过并提示。"""
    if shutil.which("docker") is None:
        return [], None
    try:
        ids = subprocess.run(
            ["docker", "ps", "-q"], capture_output=True, text=True, timeout=10
        ).stdout.split()
        if not ids:
            return [], None
        infos = json.loads(
            subprocess.run(
                ["docker", "inspect", *ids], capture_output=True, text=True, timeout=15
            ).stdout
        )
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as e:
        return [], f"docker 检查失败, 已跳过容器发现: {e}"

    insts = []
    for info in infos:
        name = (info.get("Name") or "").lstrip("/")
        image = (info.get("Config") or {}).get("Image") or ""
        if "asrflow" not in name.lower() and "asrflow" not in image.lower():
            continue
        if not info.get("State", {}).get("Running"):
            continue
        env = {}
        for kv in (info.get("Config") or {}).get("Env") or []:
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k] = v
        ws_port = int(env.get("ASR_SERVER_PORT", DEFAULT_WS_PORT))
        http_port = int(env.get("ASR_HTTP_PORT", DEFAULT_HTTP_PORT))
        note = ""
        if info.get("HostConfig", {}).get("NetworkMode") == "host":
            ws_host = http_host = "127.0.0.1"
        else:
            # bridge: 解析宿主端口映射; 未映射则退回容器 IP (linux 上宿主可直达)
            ports = (info.get("NetworkSettings") or {}).get("Ports") or {}

            def host_port(cport):
                m = ports.get(f"{cport}/tcp")
                return m[0]["HostPort"] if m else None

            ws_map, http_map = host_port(ws_port), host_port(http_port)
            if ws_map and http_map:
                ws_host = http_host = "127.0.0.1"
                ws_port, http_port = int(ws_map), int(http_map)
            else:
                cip = next(
                    (net.get("IPAddress") for net in
                     ((info.get("NetworkSettings") or {}).get("Networks") or {}).values()
                     if net.get("IPAddress")),
                    None,
                )
                if cip:
                    ws_host = http_host = cip
                    note = f"端口未映射到宿主, 直连容器 IP {cip}"
                else:
                    ws_host = http_host = None
                    note = "bridge 网络且端口未映射, 无法从宿主访问"
        if ws_host is None:
            insts.append(Instance("docker", f"docker:{name}", "?", ws_port, "?", http_port, note))
        else:
            insts.append(Instance("docker", f"docker:{name}", ws_host, ws_port, http_host, http_port, note))
    return insts, None


def find_process_instances():
    """扫描 /proc 找本仓库 main.py 进程 (后台守护进程与手工前台启动均覆盖)。"""
    if not os.path.isdir("/proc"):
        return []
    insts = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                args = [p.decode(errors="replace") for p in f.read().split(b"\0") if p]
        except OSError:
            continue
        main_idx = next((i for i, a in enumerate(args) if a.endswith("main.py")), None)
        if main_idx is None:
            continue
        main_arg = args[main_idx]
        if os.path.isabs(main_arg):
            base = os.path.dirname(os.path.abspath(main_arg))
        else:
            # 相对路径的 main.py (如容器入口 `python3 main.py`) 按目标进程自身的
            # cwd 解析; 按本脚本 cwd 解析会找不到 core/pipeline 而漏发现
            try:
                base = os.path.dirname(os.path.join(os.readlink(f"/proc/{pid}/cwd"), main_arg))
            except OSError:
                continue
        # 同级存在 core/ 与 pipeline/ 才认定是本仓库, 排除其他项目的同名 main.py
        if not (os.path.isdir(os.path.join(base, "core"))
                and os.path.isdir(os.path.join(base, "pipeline"))):
            continue

        def argval(flag, default):
            for i, a in enumerate(args):
                if a == flag and i + 1 < len(args):
                    return args[i + 1]
                if a.startswith(flag + "="):
                    return a.split("=", 1)[1]
            return default

        insts.append(
            Instance(
                "process",
                f"pid:{pid}",
                "127.0.0.1",
                int(argval("--port", DEFAULT_WS_PORT)),
                "127.0.0.1",
                int(argval("--http_port", DEFAULT_HTTP_PORT)),
            )
        )
    return insts


# ---------------------------------------------------------------------------
# HTTP 探活
# ---------------------------------------------------------------------------
def http_get_json(url, timeout):
    """返回 (状态码, json|None); 网络错误返回 (None, None)。"""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read()
            code = resp.status
    except urllib.error.HTTPError as e:
        body, code = e.read(), e.code
    except (OSError, urllib.error.URLError):
        return None, None
    try:
        return code, json.loads(body)
    except json.JSONDecodeError:
        return code, None


def probe_http(inst, timeout):
    out = {"healthz_ok": False, "healthz": None, "ready_ok": False, "reason": None}
    if inst.http_host in (None, "?"):
        out["reason"] = "HTTP 地址未知"
        return out
    code, data = http_get_json(f"http://{inst.http_host}:{inst.http_port}/healthz", timeout)
    out["healthz_ok"], out["healthz"] = code == 200 and data is not None, data
    code, data = http_get_json(f"http://{inst.http_host}:{inst.http_port}/ready", timeout)
    out["ready_ok"] = code == 200
    if code == 503 and isinstance(data, dict):
        out["reason"] = data.get("reason") or data.get("engines")
    elif code is None:
        out["reason"] = "/ready 不可达"
    return out


# ---------------------------------------------------------------------------
# WS 功能探测
# ---------------------------------------------------------------------------
def decode_audio(path, max_sec):
    """任意音频 -> 前 max_sec 秒的 16kHz 单声道 PCM16 字节流 (走 ffmpeg)。"""
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if max_sec and max_sec > 0:
        cmd += ["-t", str(max_sec)]
    cmd += ["-ac", "1", "-ar", str(TARGET_SR), "-sample_fmt", "s16", "-f", "s16le", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg 解码失败: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


async def probe_ws(inst, pcm, rate, chunk_ms, finish_timeout):
    """对实例推一段音频并等待收尾; 判定标准 = 收到任何非空识别文本。"""
    import websockets

    result = {
        "ok": False, "text": "", "finals": 0, "provisionals": 0, "partials": 0,
        "errors": [], "finished": False, "wall": 0.0,
    }
    count_keys = {"partial": "partials", "provisional": "provisionals", "final": "finals"}
    finished = asyncio.Event()
    chunk_size = int(TARGET_SR * 2 * chunk_ms / 1000)
    interval = chunk_ms / 1000.0 / rate
    t0 = time.time()

    async def receiver(ws):
        try:
            async for raw in ws:
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype in count_keys:
                    text = str(msg.get("text") or "")
                    if text:
                        if mtype == "partial":  # 滚动中间态, 只作"有内容"信号
                            result["text"] = result["text"] or text
                        else:
                            result["text"] += text
                    result[count_keys[mtype]] += 1
                elif mtype == "error":
                    result["errors"].append(msg)
                elif mtype == "session_finished":
                    result["finished"] = True
                    finished.set()
                    return
        except websockets.ConnectionClosed as e:
            # websockets>=14 (新实现) 关闭时会从 async for 抛出
            result["errors"].append({"message": f"连接被关闭: {e}"})
        except Exception as e:  # 解析等异常也计入, 不留未取回的任务异常
            result["errors"].append({"message": f"receiver: {e}"})
        else:
            # 旧版实现 (legacy, 如 12.x) 连接关闭时 async for 静默结束, 不抛异常;
            # 此处补记关闭详情, 否则只能看到误导性的"超时未收尾"
            if not result["finished"]:
                code = getattr(ws, "close_code", None) or 1006
                reason = getattr(ws, "close_reason", None) or ""
                result["errors"].append({
                    "message": f"连接在收到 session_finished 前被关闭 (code={code}"
                               + (f", reason={reason}" if reason else "") + ")"})

    try:
        async with websockets.connect(
            f"ws://{inst.ws_host}:{inst.ws_port}", open_timeout=5, close_timeout=5,
            max_size=10 * 1024 * 1024,
        ) as ws:
            await ws.send(json.dumps({
                "type": "start",
                "session_id": f"hc-{inst.ws_port}-{int(time.time())}",
                "language": "zh",
                "enable_spk": False,
            }))
            recv_task = asyncio.create_task(receiver(ws))
            await asyncio.sleep(0.3)  # 等 session_ready

            for offset in range(0, len(pcm), chunk_size):
                if finished.is_set() or recv_task.done():
                    break  # 已收尾 / 连接已被服务端关闭, 停止推流
                await ws.send(pcm[offset:offset + chunk_size])
                await asyncio.sleep(interval)

            if not finished.is_set() and not recv_task.done():
                await ws.send(json.dumps({"type": "commit"}))
                await ws.send(json.dumps({"type": "stop"}))
                # 等收尾; 连接提前断开 (receiver 结束) 也算完成等待, 不空等超时
                wait_task = asyncio.create_task(finished.wait())
                await asyncio.wait(
                    [wait_task, recv_task],
                    timeout=finish_timeout, return_when=asyncio.FIRST_COMPLETED,
                )
                for t in (wait_task, recv_task):
                    t.cancel()
                if not result["finished"] and not recv_task.done():
                    # 连接仍存活才真的是超时; 已断开时关闭详情由 receiver 记录
                    result["errors"].append({
                        "message": f"stop 后 {finish_timeout}s 未收到 session_finished (连接仍存活)"})
            else:
                recv_task.cancel()
    except (OSError, websockets.WebSocketException) as e:
        result["errors"].append({"message": f"WS 连接失败: {e}"})
        return result

    result["ok"] = bool(result["text"].strip()) and not result["errors"]
    result["wall"] = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_endpoint(raw):
    """--endpoint 解析: ws://host:port 或 host:port (可带 path, 忽略) -> (host, port)。"""
    s = raw.strip()
    if s.startswith("wss://"):
        raise ValueError("不支持 wss:// (本服务仅明文 WS), 请用 ws://host:port")
    if s.startswith("ws://"):
        s = s[len("ws://"):]
    s = s.split("/", 1)[0]
    host, _, port = s.rpartition(":")
    if not host or not port.isdigit() or not (0 < int(port) < 65536):
        raise ValueError(f"endpoint 需为 ws://host:port 形式: {raw}")
    return host, int(port)


def main():
    parser = argparse.ArgumentParser(description="ASRFlow 本机实例健康检查")
    parser.add_argument("--audio", default=os.path.join(ROOT, "fixtures", "sample_16k.wav"),
                        help="探测用音频 (默认 fixtures/sample_16k.wav)")
    # 默认 60s (>VAD_MAX_SPEECH_MS 默认 30s): 若探测音频只有 30s, 强制切分边界可能
    # 恰好落在推流结束点, 测不到 2-pass final 输出; 60s 保证中途至少触发一次强制切分
    parser.add_argument("--max-sec", type=float, default=60, help="只取音频前 N 秒 (0=完整); 默认 60s, 覆盖默认 30s 强制切分以测到 2-pass final")
    parser.add_argument("--rate", type=float, default=4.0, help="推流倍速 (越大越快)")
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--finish-timeout", type=float, default=60, help="stop 后等待收尾超时(s)")
    parser.add_argument("--http-timeout", type=float, default=3.0)
    parser.add_argument("--ws-ports", default="", help="手动指定 WS 端口列表 (逗号分隔), 跳过自动发现")
    parser.add_argument("--endpoint", default="",
                        help="直接指定 WS endpoint (ws://host:port), 跳过自动发现与 HTTP 探活, 只做功能探测")
    parser.add_argument("--list", action="store_true", help="只发现实例 + HTTP 探活, 不做 WS 功能探测")
    args = parser.parse_args()

    if args.endpoint and args.ws_ports:
        parser.error("--endpoint 与 --ws-ports 互斥")
    if args.endpoint and args.list:
        parser.error("--endpoint 模式直接做 WS 功能探测, 与 --list 互斥")

    # 发现实例
    if args.endpoint:
        try:
            host, port = parse_endpoint(args.endpoint)
        except ValueError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            sys.exit(2)
        insts = [Instance("endpoint", "endpoint", host, port, None, None,
                          "直接指定地址, 跳过发现与 HTTP 探活")]
        warn = None
    elif args.ws_ports:
        insts = [
            Instance("manual", f"manual:{p}", "127.0.0.1", int(p), "127.0.0.1", int(p) + 1,
                     "http 端口按 ws+1 推测")
            for p in args.ws_ports.split(",") if p.strip()
        ]
        warn = None
    else:
        dinsts, warn = find_docker_instances()
        pinsts = find_process_instances()
        seen, insts = set(), []
        # 同端口理论上 docker 与进程不会并存; 万一重复保留先出现的 (docker 优先)
        for i in dinsts + pinsts:
            if i.key in seen:
                continue
            seen.add(i.key)
            insts.append(i)

    if not insts:
        print("[ERROR] 未发现本机 asrflow 实例 (docker 容器 / main.py 进程)。" if not warn
              else f"[WARN] {warn}")
        if not insts:
            sys.exit(1)

    print("=" * 70)
    print(f"发现 {len(insts)} 个 asrflow 实例" + (" (--list: 仅 HTTP 探活)" if args.list else ""))
    for i, inst in enumerate(insts, 1):
        print(f"  [{i}/{len(insts)}] {inst.label()}  来源={inst.source}"
              + (f"  [{inst.note}]" if inst.note else ""))
    if warn:
        print(f"  [WARN] {warn}")
    print("=" * 70)

    # 功能探测准备 (仅非 --list 模式需要)
    pcm = b""
    if not args.list:
        if not os.path.exists(args.audio):
            print(f"[ERROR] 探测音频不存在: {args.audio}", file=sys.stderr)
            sys.exit(2)
        if shutil.which("ffmpeg") is None:
            print("[ERROR] 未找到 ffmpeg (mp3 解码需要)", file=sys.stderr)
            sys.exit(2)
        try:
            import websockets  # noqa: F401
        except ImportError:
            venv_py = os.path.join(ROOT, ".venv", "bin", "python")
            if os.path.exists(venv_py) and os.path.abspath(venv_py) != sys.executable:
                print(f"[INFO] 当前解释器缺 websockets, 改用 {venv_py}")
                os.execv(venv_py, [venv_py, os.path.abspath(__file__)] + sys.argv[1:])
            print("[ERROR] 缺 websockets 依赖 (先装 requirements.txt 或用 .venv 解释器)", file=sys.stderr)
            sys.exit(2)
        pcm = decode_audio(args.audio, args.max_sec)
        print(f"探测音频: {args.audio} 前 {args.max_sec}s -> "
              f"{len(pcm) / 2 / TARGET_SR:.1f}s PCM16, 倍速 {args.rate}x\n")

    all_ok = True
    for i, inst in enumerate(insts, 1):
        print(f"[{i}/{len(insts)}] {inst.label()}")
        if inst.ws_host == "?":
            # bridge 网络且端口未映射的容器: 已发现但无法从宿主探测
            print(f"  [FAIL] 无法从宿主访问: {inst.note or '地址未知'}")
            all_ok = False
            print()
            continue
        if inst.source == "endpoint":
            h = None
            print("  HTTP : 跳过 (--endpoint 直接探测指定地址)")
        else:
            h = probe_http(inst, args.http_timeout)
            hz = h["healthz"] or {}
            hz_desc = (f"uptime {hz.get('uptime_seconds', '?')}s, sessions {hz.get('active_sessions', '?')}"
                       if h["healthz_ok"] else "不可达")
            print(f"  HTTP : healthz={'OK' if h['healthz_ok'] else 'FAIL'} ({hz_desc}) | "
                  f"ready={'OK' if h['ready_ok'] else 'NOT READY'}"
                  + (f" ({h['reason']})" if h["reason"] else ""))

        if args.list:
            if not (h["healthz_ok"] and h["ready_ok"]):
                all_ok = False
            print()
            continue

        r = asyncio.run(probe_ws(inst, pcm, args.rate, args.chunk_ms, args.finish_timeout))
        verdict = "PASS" if r["ok"] else "FAIL"
        print(f"  WS   : {verdict} — 文本 {len(r['text'])} 字 "
              f"(final {r['finals']}, provisional {r['provisionals']}, partial {r['partials']}), "
              f"{r['wall']:.1f}s, 收尾={'正常' if r['finished'] else '未收尾'}")
        if r["text"]:
            preview = r["text"][:60] + ("..." if len(r["text"]) > 60 else "")
            print(f"         预览: {preview}")
        if r["errors"]:
            print(f"         错误: {r['errors']}")
        # 判定以功能为主: ready 未就绪但能出文本记 WARN, 不翻转整体结果
        if not r["ok"]:
            all_ok = False
        elif h is not None and not h["ready_ok"]:
            print("         [WARN] /ready 未就绪但功能可用 (引擎重载中?), 请关注")
        print()

    print("=" * 70)
    print("健康检查结果:", "ALL PASS" if all_ok else "存在 FAIL 实例")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
