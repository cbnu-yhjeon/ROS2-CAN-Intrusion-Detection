"""온라인/오프라인 판정 평가.

- :func:`align` : 노드가 남긴 판정 로그를 원본 세그먼트 행에 맞춘다. 전송 중 유실이 있어도
  순서는 보존되므로 (can_id, payload)가 같은 다음 행을 앞에서부터 탐욕적으로 찾는다.
- :func:`classification` : precision / recall / F1
- :func:`detection_delays` : 공격 구간(burst)마다 첫 주입 메시지 수신 → 첫 경보 판정
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from can_ids.dataset import iter_frames
from can_ids.engine import DetectionEngine, run_stream
from can_ids.features import compute_features_offline


def payload_hex(df: pd.DataFrame) -> np.ndarray:
    data = df[[f"data_{i}" for i in range(8)]].to_numpy(dtype=np.uint8)
    return np.array([bytes(r[:d]).hex() for r, d in zip(data, df["dlc"].to_numpy())], dtype=object)


def align(segment: pd.DataFrame, verdicts: pd.DataFrame, lookahead: int = 5000) -> np.ndarray:
    """verdict 각 행에 대응하는 segment 행 번호 (못 찾으면 -1)."""
    seg_key = list(zip(segment["can_id"].to_numpy(), payload_hex(segment)))
    v_data = verdicts["data"].fillna("").astype(str).to_numpy()
    v_key = list(zip(verdicts["can_id"].to_numpy(), v_data))
    out = np.full(len(v_key), -1, dtype=np.int64)
    j, n = 0, len(seg_key)
    for i, k in enumerate(v_key):
        stop = min(n, j + lookahead)
        for jj in range(j, stop):
            if seg_key[jj] == k:
                out[i] = jj
                j = jj + 1
                break
    return out


def classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, bool)
    y_pred = np.asarray(y_pred, bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else 0.0}


def bursts(labels: np.ndarray, stamps: np.ndarray, gap_s: float = 0.5):
    """연속된 공격 행을 시간 간격 gap_s 이상에서 끊어 (시작 행, 끝 행) 목록."""
    idx = np.flatnonzero(labels)
    if len(idx) == 0:
        return []
    cut = np.flatnonzero(np.diff(stamps[idx]) > gap_s)
    starts = np.r_[idx[0], idx[cut + 1]]
    ends = np.r_[idx[cut], idx[-1]]
    return list(zip(starts, ends))


def detection_delays(labels: np.ndarray, stamps: np.ndarray, preds: np.ndarray, t_done: np.ndarray,
                     gap_s: float = 0.5) -> Dict[str, float]:
    """행 단위 배열 (수신된 행만) 기준. 각 burst의 첫 공격 행 수신 시각 → 그 burst 안 첫 경보 판정 시각."""
    delays, missed = [], 0
    for s, e in bursts(labels, stamps, gap_s):
        hit = np.flatnonzero(preds[s:e + 1])
        if len(hit) == 0:
            missed += 1
            continue
        delays.append((t_done[s + hit[0]] - stamps[s]) * 1e3)
    d = np.array(delays) if delays else np.array([np.nan])
    n = len(delays) + missed
    return {"bursts": n, "missed_bursts": missed,
            "delay_p50_ms": float(np.nanpercentile(d, 50)) if delays else float("nan"),
            "delay_max_ms": float(np.nanmax(d)) if delays else float("nan")}


def offline_predictions(segment: pd.DataFrame, detector) -> np.ndarray:
    """오프라인 배치 경로: pandas 특징 → 한 번에 추론."""
    detector.reset()
    _, preds = detector.score(compute_features_offline(segment))
    return preds


def streaming_predictions(segment: pd.DataFrame, detector, batch_size: int = 1) -> np.ndarray:
    """프로세스 안 스트리밍 경로 (원본 타임스탬프): 상태 특징 추출기 + 엔진."""
    detector.reset()
    vs = run_stream(DetectionEngine(detector), iter_frames(segment), batch_size)
    return np.array([v.is_attack for v in vs])


def evaluate_online(segment: pd.DataFrame, verdicts: pd.DataFrame,
                    offline_pred: Optional[np.ndarray] = None) -> Dict[str, float]:
    """노드 판정 로그 → 지표 묶음."""
    m = align(segment, verdicts)
    ok = m >= 0
    v = verdicts[ok]
    rows = m[ok]
    y = segment["label"].to_numpy()[rows]
    pred = v["pred"].to_numpy().astype(bool)
    lat = (v["t_done"].to_numpy() - v["stamp"].to_numpy()) * 1e3
    res = {"frames_sent": len(segment), "frames_judged": int(len(verdicts)),
           "frames_lost": int(len(segment) - ok.sum()), "unmatched": int((~ok).sum())}
    res.update({f"online_{k}": val for k, val in classification(y, pred).items()})
    res.update({
        "latency_p50_ms": float(np.percentile(lat, 50)), "latency_p99_ms": float(np.percentile(lat, 99)),
        "latency_max_ms": float(lat.max()),
        "mean_batch": float(v["batch"].mean()),
    })
    dur = v["stamp"].iloc[-1] - v["stamp"].iloc[0] if len(v) > 1 else float("nan")
    res["throughput_fps"] = float(len(v) / dur) if dur else float("nan")
    res.update(detection_delays(y, v["stamp"].to_numpy(), pred, v["t_done"].to_numpy()))
    if offline_pred is not None:
        res.update({f"offline_{k}": val for k, val in
                    classification(segment["label"].to_numpy(), offline_pred).items()})
        res["f1_gap_pp"] = 100 * abs(res["online_f1"] - res["offline_f1"])
        res["verdict_agreement"] = float((offline_pred[rows] == pred).mean())
    return res
