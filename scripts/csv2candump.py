#!/usr/bin/env python3
"""HCRL CSV → candump 로그 (canplayer 입력) + 정답 라벨 세그먼트 CSV.

    python scripts/csv2candump.py data/Data/raw/DoS_dataset.csv -o results/segments/dos \
        --start 0 --duration 60

출력:
  <out>.log  : ``(1478198376.389427) vcan0 316#05216809212100``
  <out>.csv  : 정규화된 세그먼트 (timestamp, can_id, dlc, data_0..7, label) — 평가 기준
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "can_ids"))

from can_ids.dataset import load_hcrl_csv, select_window, write_segment  # noqa: E402


def to_candump_lines(df, interface: str = "vcan0"):
    data = df[[f"data_{i}" for i in range(8)]].to_numpy()
    for t, cid, dlc, row in zip(df["timestamp"].to_numpy(), df["can_id"].to_numpy(),
                                df["dlc"].to_numpy(), data):
        payload = "".join(f"{b:02X}" for b in row[:dlc])
        cid_s = f"{cid:03X}" if cid <= 0x7FF else f"{cid:08X}"
        yield f"({t:.6f}) {interface} {cid_s}#{payload}\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True, help="출력 경로 (확장자 없이)")
    ap.add_argument("--start", type=float, default=0.0, help="로그 시작 기준 초")
    ap.add_argument("--duration", type=float, default=None, help="구간 길이(초), 생략하면 끝까지")
    ap.add_argument("--max-rows", type=int, default=None, help="원본에서 읽을 최대 행 수")
    ap.add_argument("-i", "--interface", default="vcan0")
    args = ap.parse_args()

    nrows = args.max_rows
    if nrows is None and args.duration is not None:
        # 최대 메시지율(DoS 구간 ~4.3 kHz)을 넉넉히 잡아 필요한 행만 읽는다
        nrows = int((args.start + args.duration) * 5000) + 10000
    df = select_window(load_hcrl_csv(args.csv, nrows=nrows), args.start, args.duration)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out.with_suffix(".log"), "w") as f:
        f.writelines(to_candump_lines(df, args.interface))
    write_segment(df, args.out.with_suffix(".csv"))
    dur = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    print(f"{args.out}.log: {len(df)} frames, {dur:.1f} s, attack={int(df['label'].sum())}")


if __name__ == "__main__":
    main()
