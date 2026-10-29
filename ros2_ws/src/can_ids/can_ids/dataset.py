"""HCRL Car-Hacking CSV 로더.

원본 형식: ``timestamp,can_id(hex),dlc,b0..b{dlc-1}(hex),flag``
DLC < 8 인 행은 필드 수가 적어서 flag 위치가 달라진다. 원 저장소 로더는 이 경우
flag를 NaN으로 읽어 정상(0)으로 처리했는데, 공격(T) 행 중 DLC < 8 은 없어서
라벨 결과는 같다. 여기서는 flag를 ``3 + dlc`` 위치에서 정확히 읽는다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from can_ids.features import CanFrame

COLUMNS = ["timestamp", "can_id", "dlc"] + [f"data_{i}" for i in range(8)] + ["label"]


def load_hcrl_csv(path, skiprows: int = 0, nrows: Optional[int] = None) -> pd.DataFrame:
    """원본 CSV → timestamp(float), can_id(int), dlc, data_0..7(int), label(0/1)."""
    raw = pd.read_csv(
        path, header=None, names=list(range(12)), dtype=str,
        skiprows=skiprows, nrows=nrows, engine="c",
    )
    dlc = raw[2].astype(int).to_numpy()
    cells = raw.to_numpy(dtype=object)
    n = len(raw)
    flag = cells[np.arange(n), np.clip(3 + dlc, 3, 11)]
    out = pd.DataFrame({
        "timestamp": raw[0].astype(float).to_numpy(),
        "can_id": raw[1].map(lambda x: int(x, 16)).to_numpy(),
        "dlc": dlc,
    })
    for i in range(8):
        col = raw[3 + i].fillna("00")
        vals = col.map(lambda x: int(x, 16) if x not in ("R", "T") else 0).to_numpy()
        out[f"data_{i}"] = np.where(i < dlc, vals, 0)
    out["label"] = (flag == "T").astype(int)
    return out


def select_window(df: pd.DataFrame, start_s: float = 0.0, duration_s: Optional[float] = None) -> pd.DataFrame:
    """로그 시작 기준 [start_s, start_s + duration_s) 구간."""
    t0 = df["timestamp"].iloc[0]
    rel = df["timestamp"] - t0
    mask = rel >= start_s
    if duration_s is not None:
        mask &= rel < start_s + duration_s
    return df[mask].reset_index(drop=True)


def iter_frames(df: pd.DataFrame) -> Iterator[CanFrame]:
    data = df[[f"data_{i}" for i in range(8)]].to_numpy(dtype=np.uint8)
    for t, cid, dlc, row, lab in zip(df["timestamp"].to_numpy(), df["can_id"].to_numpy(),
                                     df["dlc"].to_numpy(), data, df["label"].to_numpy()):
        yield CanFrame(float(t), int(cid), int(dlc), bytes(row[:dlc]), int(lab))


def write_segment(df: pd.DataFrame, path) -> None:
    """정답 라벨이 있는 정규화된 세그먼트 CSV (평가 기준)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")


def read_segment(path) -> pd.DataFrame:
    return pd.read_csv(path)
