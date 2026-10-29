#!/usr/bin/env python3
"""온라인 실험 실행기: vcan0 재생 → ROS 2 탐지 노드 → 판정 로그 → 지표.

vcan0이 있는 네트워크 네임스페이스 안에서 실행해야 한다 (root가 없으면 in_netns.sh 사용):

    scripts/in_netns.sh .venv/bin/python scripts/run_online.py --suite main
    scripts/in_netns.sh .venv/bin/python scripts/run_online.py --segments dos --models xgb --batch-ms 0

실행마다 results/runs/<run_id>/ 에 판정 로그·노드 로그·metrics.json을 남기고,
모은 지표는 results/online_<suite>.json 에 저장한다.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "can_ids"))

from can_ids.dataset import read_segment  # noqa: E402
from can_ids.evaluate import evaluate_online, offline_predictions  # noqa: E402
from can_ids.models import load_detector  # noqa: E402

DRAIN_LIMIT_S = 120
SEGMENTS = ["dos", "fuzzy", "rpm", "gear", "normal"]

# ID: (모델, 마이크로 배치 창 ms)
SUITES = {
    "main": [
        ("R1", "xgb", 0.0), ("R2", "iforest", 0.0), ("R3", "lstm", 0.0),
        ("R4", "xgb", 1.0), ("R5", "ensemble", 1.0),
        ("R2b", "iforest", 1.0), ("R3b", "lstm", 1.0),
    ],
}
THROUGHPUT_SPEEDS = [1, 2, 4, 8, 16]
THROUGHPUT_CONFIGS = [("R1", "xgb", 0.0), ("R4", "xgb", 1.0), ("R2b", "iforest", 1.0),
                      ("R3b", "lstm", 1.0), ("R5", "ensemble", 1.0)]


def ros_env() -> dict:
    env = dict(os.environ)
    env.setdefault("ROS_LOCALHOST_ONLY", "1")
    env.setdefault("ROS_DOMAIN_ID", "42")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def shell_with_ros(cmd: str) -> list:
    setup = f"source /opt/ros/humble/setup.bash && source {ROOT}/ros2_ws/install/setup.bash"
    venv = ROOT / ".venv/bin/activate"
    if venv.exists():
        setup += f" && source {venv}"
    return ["bash", "-c", f"{setup} && exec {cmd}"]


def proc_stats(pid: int):
    """(utime+stime 초, RSS MB)"""
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().rsplit(")", 1)[1].split()
        tick = os.sysconf("SC_CLK_TCK")
        cpu = (int(parts[11]) + int(parts[12])) / tick
        with open(f"/proc/{pid}/statm") as f:
            rss = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
        return cpu, rss
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return None


def find_pid(pattern: str):
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    return int(out[-1]) if out else None


def count_lines(path: Path) -> int:
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f) - 1
    except FileNotFoundError:
        return 0


def run_once(run_dir: Path, segment: str, model: str, batch_ms: float, speed: float = 1.0,
             offline_pred=None, input_mode: str = "topic") -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    seg_csv = ROOT / f"data/segments/{segment}.csv"
    seg_log = ROOT / f"data/segments/{segment}.log"
    verdicts = run_dir / "verdicts.csv"
    node_log = open(run_dir / "launch.log", "w")
    launch = subprocess.Popen(
        shell_with_ros(f"ros2 launch can_ids ids.launch.py model:={model} batch_window_ms:={batch_ms} "
                       f"verdict_log:={verdicts} model_dir:={ROOT / 'models'} input:={input_mode}"),
        stdout=node_log, stderr=subprocess.STDOUT, env=ros_env(), start_new_session=True)
    try:
        # 노드 준비 대기
        deadline = time.time() + 60
        while time.time() < deadline:
            if "model=" in (run_dir / "launch.log").read_text() and verdicts.exists():
                break
            if launch.poll() is not None:
                raise RuntimeError(f"launch 종료: {(run_dir / 'launch.log').read_text()[-2000:]}")
            time.sleep(0.2)
        time.sleep(2.0)  # DDS 매칭
        pid = find_pid(f"ids_node.*{verdicts}") or find_pid("lib/can_ids/ids_node")
        samples = []
        stop = threading.Event()

        def sampler():
            while not stop.is_set():
                s = proc_stats(pid) if pid else None
                if s:
                    samples.append((time.time(), *s))
                time.sleep(0.5)

        th = threading.Thread(target=sampler, daemon=True)
        th.start()
        t_play = time.time()
        player = subprocess.run(
            [sys.executable, str(ROOT / "scripts/pycanplayer.py"), "-I", str(seg_log), "-i", "vcan0",
             "--speed", str(speed)], capture_output=True, text=True)
        play_s = time.time() - t_play
        # 처리 대기: 판정 수가 3초 동안 늘지 않으면 종료
        last, stable_since = -1, time.time()
        # 과부하로 밀린 노드가 끝없이 처리하는 경우를 막기 위해 재생 후 최대 DRAIN_LIMIT_S초만 기다린다
        drain_timeout = False
        while time.time() - stable_since < 3.0:
            if time.time() - t_play > play_s + DRAIN_LIMIT_S:
                drain_timeout = True
                break
            n = count_lines(verdicts)
            if n != last:
                last, stable_since = n, time.time()
            time.sleep(0.5)
        stop.set()
        th.join()
    finally:
        os.killpg(launch.pid, signal.SIGINT)
        try:
            launch.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(launch.pid, signal.SIGKILL)
        node_log.close()

    seg = read_segment(seg_csv)
    v = pd.read_csv(verdicts, dtype={"data": str})
    res = {"segment": segment, "model": model, "batch_window_ms": batch_ms, "speed": speed, "input": input_mode,
           "player": player.stdout.strip(), "drain_timeout": drain_timeout}
    res.update(evaluate_online(seg, v, offline_pred))
    if len(samples) >= 2:
        (t0, c0, _), (t1, c1, _) = samples[0], samples[-1]
        res["cpu_pct"] = 100 * (c1 - c0) / max(t1 - t0, 1e-9)
        res["rss_mb"] = max(s[2] for s in samples)
    (run_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    return res


def fmt(r: dict) -> str:
    return (f"{r['input']:9s} {r['segment']:6s} {r['model']:8s} batch={r['batch_window_ms']:g}ms x{r['speed']:g} | "
            f"lost={r['frames_lost']} F1 on={r['online_f1']:.4f} off={r.get('offline_f1', float('nan')):.4f} | "
            f"lat p50={r['latency_p50_ms']:.2f} p99={r['latency_p99_ms']:.2f} ms | "
            f"delay p50={r['delay_p50_ms']:.1f} ms | cpu={r.get('cpu_pct', 0):.0f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["main", "throughput", "custom"], default="custom")
    ap.add_argument("--segments", default=",".join(SEGMENTS))
    ap.add_argument("--models", default="xgb")
    ap.add_argument("--batch-ms", type=float, default=0.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--input", default="topic", help="topic,socketcan (쉼표로 여러 개)")
    args = ap.parse_args()

    try:
        socket.if_nametoindex("vcan0")
    except OSError:
        sys.exit("vcan0이 없다: scripts/setup_vcan.sh 또는 scripts/in_netns.sh 안에서 실행할 것")

    inputs = args.input.split(",")
    if args.suite == "main":
        plan = [(rid, seg, m, b, 1.0, inp) for inp in inputs for seg in args.segments.split(",")
                for rid, m, b in SUITES["main"]]
    elif args.suite == "throughput":
        plan = [(rid, "dos", m, b, s, inp) for inp in inputs for rid, m, b in THROUGHPUT_CONFIGS
                for s in THROUGHPUT_SPEEDS]
    else:
        plan = [("custom", seg, m, args.batch_ms, args.speed, inp) for inp in inputs
                for seg in args.segments.split(",") for m in args.models.split(",")]

    offline_cache = {}
    results = []
    out = ROOT / f"results/online_{args.suite}.json"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for i, (rid, seg, model, batch_ms, speed, inp) in enumerate(plan):
        key = (seg, model)
        if key not in offline_cache:
            offline_cache[key] = offline_predictions(read_segment(ROOT / f"data/segments/{seg}.csv"),
                                                     load_detector(model, ROOT / "models"))
        run_dir = ROOT / f"results/runs/{stamp}/{inp}_{rid}_{seg}_{model}_b{batch_ms:g}_x{speed:g}"
        r = run_once(run_dir, seg, model, batch_ms, speed, offline_cache[key], inp)
        r["id"] = rid
        results.append(r)
        print(f"[{i + 1}/{len(plan)}] {rid:4s} {fmt(r)}", flush=True)
        out.write_text(json.dumps(results, indent=2))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
