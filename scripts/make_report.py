#!/usr/bin/env python3
"""results/*.json → results/REPORT.md (실험 결과 표)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ATTACKS = ["dos", "fuzzy", "rpm", "gear"]
CONFIG_ORDER = ["R1", "R2", "R2b", "R3", "R3b", "R4", "R5"]
CONFIG_DESC = {
    "R1": "XGBoost · 메시지 단위", "R2": "Isolation Forest · 메시지 단위", "R2b": "Isolation Forest · 1 ms 배치",
    "R3": "LSTM-AE · 메시지 단위", "R3b": "LSTM-AE · 1 ms 배치", "R4": "XGBoost · 1 ms 배치",
    "R5": "앙상블(XGB→LSTM 확인) · 1 ms 배치",
}


def load(name):
    p = RES / name
    return pd.DataFrame(json.loads(p.read_text())) if p.exists() else None


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def f(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def order(df):
    return df.assign(_o=df["id"].map({k: i for i, k in enumerate(CONFIG_ORDER)})).sort_values("_o")


def section_benchmark():
    p = RES / "benchmark_models.json"
    if not p.exists():
        return ""
    b = json.loads(p.read_text())
    rows = pd.DataFrame(b["rows"])
    piv = rows.pivot_table(index=["model", "backend"], columns="batch", values="per_msg_us").round(1)
    piv.columns = [f"배치 {c} (µs/msg)" for c in piv.columns]
    piv = piv.reset_index()
    return (f"## 1. 모델 추론 비용 (오프라인, 단일 스레드)\n\n"
            f"스트리밍 특징 추출: **{b['feature_us_per_msg']:.2f} µs/msg**. 아래는 메시지당 평균 추론 시간.\n\n"
            f"{md(piv)}\n")


def section_main(df, title):
    df = order(df)
    lines = [f"## {title}\n"]
    # F1: 온라인 / 오프라인
    t = []
    for rid, g in df.groupby("id", sort=False):
        row = {"ID": rid, "구성": CONFIG_DESC.get(rid, rid)}
        for seg in ATTACKS:
            r = g[g.segment == seg]
            if len(r):
                r = r.iloc[0]
                row[seg] = f"{r.online_f1:.4f} / {r.offline_f1:.4f}"
        r = g[g.segment == "normal"]
        if len(r):
            row["normal FPR"] = f"{100 * r.iloc[0].online_fpr:.2f}% / {100 * r.iloc[0].offline_fpr:.2f}%"
        row["최대 |ΔF1| (%p)"] = f(g.f1_gap_pp.max())
        t.append(row)
    lines.append("### F1 — 온라인 / 오프라인 (같은 60초 로그)\n\n" + md(pd.DataFrame(t)) + "\n")
    # 지연 · 유실 · 자원
    t = []
    for rid, g in df.groupby("id", sort=False):
        att = g[g.segment.isin(ATTACKS)]
        t.append({
            "ID": rid, "구성": CONFIG_DESC.get(rid, rid),
            "유실 프레임": int(g.frames_lost.sum()),
            "종단 지연 p50 (ms)": f(g.latency_p50_ms.median()),
            "종단 지연 p99 (ms, 최악 세그먼트)": f(g.latency_p99_ms.max()),
            "탐지 지연 p50 (ms)": f(att.delay_p50_ms.median(), 1),
            "놓친 공격 구간": f"{int(att.missed_bursts.sum())}/{int(att.bursts.sum())}",
            "평균 배치": f(g.mean_batch.mean(), 1),
            "CPU %": f(g.cpu_pct.mean(), 0) if "cpu_pct" in g else "–",
            "RSS MB": f(g.rss_mb.max(), 0) if "rss_mb" in g else "–",
        })
    lines.append("### 지연시간 · 유실 · 자원\n\n" + md(pd.DataFrame(t)) + "\n")
    # DoS 세그먼트 상세
    d = df[df.segment == "dos"]
    if len(d):
        t = [{"ID": r.id, "p50": f(r.latency_p50_ms), "p99": f(r.latency_p99_ms), "max": f(r.latency_max_ms, 1),
              "처리량 (msg/s)": f(r.throughput_fps, 0), "유실": int(r.frames_lost),
              "탐지 지연 p50 (ms)": f(r.delay_p50_ms, 1)} for r in d.itertuples()]
        lines.append("### DoS 세그먼트 상세 (지연 단위 ms)\n\n" + md(pd.DataFrame(t)) + "\n")
    return "\n".join(lines)


def section_throughput(df):
    df = order(df)
    seg_frames = df.frames_sent.iloc[0]
    t = []
    for rid, g in df.groupby("id", sort=False):
        row = {"ID": rid, "구성": CONFIG_DESC.get(rid, rid)}
        best = None
        for r in g.sort_values("speed").itertuples():
            offered = seg_frames / (60.0 / r.speed)
            ok = r.frames_lost == 0 and r.latency_p99_ms < 50 and not getattr(r, "drain_timeout", False)
            row[f"x{r.speed:g} ({offered:,.0f}/s)"] = (
                f"{'✅' if ok else '❌'} 유실 {100 * r.frames_lost / r.frames_sent:.1f}% · p99 {f(r.latency_p99_ms, 1)}")
            if ok:
                best = offered
        row["유실 없이 처리한 최대 평균 입력률"] = f"{best:,.0f} msg/s" if best else "–"
        t.append(row)
    return ("## 3. 처리량 — DoS 세그먼트 배속 재생 (직접 SocketCAN 입력)\n\n"
            "성공 = 유실 0 이고 종단 지연 p99 < 50 ms. 괄호는 평균 입력률 (DoS 버스트 구간 순간 입력률은 더 높음).\n\n"
            + md(pd.DataFrame(t)) + "\n")


def section_transport(sock, topic, topic_x2):
    s = sock[sock.segment == "dos"].set_index("id")
    t = []
    for r in order(topic).itertuples():
        if r.id not in s.index:
            continue
        o = s.loc[r.id]
        t.append({"ID": r.id, "구성": CONFIG_DESC.get(r.id, r.id),
                  "유실 socketcan / topic": f"{int(o.frames_lost)} / {int(r.frames_lost)}",
                  "p50 ms socketcan / topic": f"{f(o.latency_p50_ms)} / {f(r.latency_p50_ms)}",
                  "p99 ms socketcan / topic": f"{f(o.latency_p99_ms)} / {f(r.latency_p99_ms)}",
                  "F1 socketcan / topic": f"{o.online_f1:.4f} / {r.online_f1:.4f}"})
    out = ("## 4. 입력 경로 비교 — 직접 SocketCAN vs ROS 토픽 (/from_can_bus, DoS 1배속)\n\n"
           "topic 경로 = socketcan_bridge(파이썬) → DDS → ids_node 구독 콜백.\n\n" + md(pd.DataFrame(t)) + "\n")
    if topic_x2 is not None and len(topic_x2):
        r = topic_x2.iloc[0]
        out += (f"\ntopic 경로 XGBoost 2배속: 유실 {int(r.frames_lost)}/{int(r.frames_sent)}, "
                f"p99 {f(r.latency_p99_ms, 1)} ms{' (처리 대기 시간 초과)' if r.get('drain_timeout') else ''}.\n")
    return out


def main():
    parts = ["# 실험 결과\n",
             "자동 생성: `python scripts/make_report.py`. 데이터: HCRL Car-Hacking, 각 로그 앞 60초 "
             "(공격 로그는 약 1초 뒤부터 4~5초 공격 / 3초 휴지 반복). 실행 환경: WSL2, 12코어, "
             "사용자 네트워크 네임스페이스 안 vcan0, 파이썬 재생기(pycanplayer).\n"]
    parts.append(section_benchmark())
    sock = load("online_main_socketcan.json")
    if sock is not None:
        parts.append(section_main(sock, "2. 온라인 탐지 (vcan0 → ids_node 직접 SocketCAN 입력, 1배속)"))
    thr = load("online_throughput_socketcan.json")
    if thr is not None:
        parts.append(section_throughput(thr))
    topic = load("online_main_topic.json")
    if sock is not None and topic is not None:
        parts.append(section_transport(sock, topic, load("online_topic_x2.json")))
    (RES / "REPORT.md").write_text("\n".join(parts))
    print(f"→ {RES / 'REPORT.md'}")


if __name__ == "__main__":
    main()
