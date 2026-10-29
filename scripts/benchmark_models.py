#!/usr/bin/env python3
"""모델 추론 비용 마이크로 벤치마크 (ROS 없이, 단일 스레드).

메시지 1개 단위 호출과 배치 호출의 메시지당 시간을 비교한다.
특징 추출(스트리밍) 비용도 따로 잰다.

    python scripts/benchmark_models.py --segment data/segments/dos.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "can_ids"))

from can_ids.dataset import iter_frames, read_segment  # noqa: E402
from can_ids.features import StreamingFeatureExtractor, compute_features_offline  # noqa: E402
from can_ids.models import load_detector  # noqa: E402


def per_call_ms(fn, X, batch, n_calls):
    times = []
    for k in range(n_calls):
        s = (k * batch) % (len(X) - batch)
        xb = X[s:s + batch]
        t = time.perf_counter()
        fn(xb)
        times.append((time.perf_counter() - t) * 1e3)
    return np.array(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default=ROOT / "data/segments/dos.csv", type=Path)
    ap.add_argument("--models", default="xgb,iforest,lstm,ensemble")
    ap.add_argument("--batches", default="1,8,32,128,512")
    ap.add_argument("--out", default=ROOT / "results/benchmark_models.json", type=Path)
    args = ap.parse_args()

    seg = read_segment(args.segment)
    X = compute_features_offline(seg)

    fx = StreamingFeatureExtractor()
    frames = list(iter_frames(seg.iloc[:20000]))
    t = time.perf_counter()
    for f in frames:
        fx.update_frame(f)
    feat_us = (time.perf_counter() - t) / len(frames) * 1e6
    print(f"streaming feature extraction: {feat_us:.2f} µs/msg")

    rows = []
    for name in args.models.split(","):
        backends = {"lstm": ["onnx", "torchscript"], "iforest": ["onnx"], "xgb": ["native"]}.get(name, ["onnx"])
        for backend in backends:
            det = load_detector(name, ROOT / "models", backend="onnx" if backend == "native" else backend, threads=1)
            det.score(X[:600])  # 워밍업 + LSTM 이력 채우기
            for b in map(int, args.batches.split(",")):
                n_calls = max(20, min(2000, 20000 // b))
                ms = per_call_ms(det.score, X, b, n_calls)
                per_msg_us = ms.mean() / b * 1e3
                row = {"model": name, "backend": backend, "batch": b,
                       "call_p50_ms": float(np.percentile(ms, 50)), "call_p99_ms": float(np.percentile(ms, 99)),
                       "per_msg_us": float(per_msg_us), "max_fps": float(1e6 / per_msg_us)}
                rows.append(row)
                print(f"{name:9s} {backend:11s} batch={b:4d}  call p50={row['call_p50_ms']:.3f} ms "
                      f"p99={row['call_p99_ms']:.3f} ms  → {per_msg_us:8.1f} µs/msg  ({row['max_fps']:,.0f} msg/s)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"feature_us_per_msg": feat_us, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
