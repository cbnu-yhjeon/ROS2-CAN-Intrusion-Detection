#!/usr/bin/env python3
"""can-utils ``canplayer``의 파이썬 대체 (표준 라이브러리만 사용).

candump 로그를 원래 시간 간격대로 SocketCAN 인터페이스에 보낸다.
sleep 오차를 줄이려고 마지막 0.5 ms는 busy-wait 한다.

    python scripts/pycanplayer.py -I results/segments/dos.log -i vcan0 --speed 1
"""
from __future__ import annotations

import argparse
import errno
import socket
import struct
import time

import numpy as np

CAN_EFF_FLAG = 0x80000000


def parse_candump(path):
    stamps, frames = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ts, _iface, body = line.split()
            cid_s, payload = body.split("#", 1)
            cid = int(cid_s, 16)
            if len(cid_s) > 3:
                cid |= CAN_EFF_FLAG
            data = bytes.fromhex(payload)
            stamps.append(float(ts.strip("()")))
            frames.append(struct.pack("=IB3x8s", cid, len(data), data.ljust(8, b"\0")))
    return np.array(stamps), frames


def play(stamps, frames, interface: str, speed: float = 1.0, spin_s: float = 0.0005):
    sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.bind((interface,))
    rel = (stamps - stamps[0]) / speed
    lateness = np.empty(len(frames))
    start = time.perf_counter()
    for i, (r, fr) in enumerate(zip(rel, frames)):
        target = start + r
        while True:
            now = time.perf_counter()
            remaining = target - now
            if remaining <= 0:
                break
            if remaining > spin_s:
                time.sleep(remaining - spin_s)
        while True:
            try:
                sock.send(fr)
                break
            except OSError as e:  # 송신 큐가 가득 차면 잠깐 기다렸다 재시도
                if e.errno != errno.ENOBUFS:
                    raise
                time.sleep(0.0001)
        lateness[i] = time.perf_counter() - target
    sock.close()
    return lateness


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-I", "--infile", required=True)
    ap.add_argument("-i", "--interface", default="vcan0")
    ap.add_argument("--speed", type=float, default=1.0, help="재생 배속 (2 = 두 배 빠르게)")
    args = ap.parse_args()
    stamps, frames = parse_candump(args.infile)
    t0 = time.time()
    late = play(stamps, frames, args.interface, args.speed) * 1e3
    wall = time.time() - t0
    print(f"sent {len(frames)} frames in {wall:.2f} s ({len(frames) / wall:.0f} fps); "
          f"send lateness ms p50={np.percentile(late, 50):.3f} p99={np.percentile(late, 99):.3f} "
          f"max={late.max():.3f}")


if __name__ == "__main__":
    main()
