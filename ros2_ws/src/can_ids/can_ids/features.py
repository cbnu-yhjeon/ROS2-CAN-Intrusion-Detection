"""CAN 특징 추출기 — 오프라인(배치)과 온라인(스트리밍) 두 경로.

원 저장소(PROJECT-CAN-Anomaly-Detection)의 전처리를 그대로 옮겼다.

- 바이트: DLC 이후 바이트는 0으로 채운다 (``normalize_bytes_and_dlc``)
- ``delta_t``    : 직전 메시지(전체 ID)와의 시각 차, 첫 메시지는 0
- ``delta_t_id`` : 같은 ID 직전 메시지와의 시각 차, ID별 첫 메시지는 0
- 두 시간 특징은 학습 데이터의 99 분위수에서 상한 클리핑 (학습 로그 값 고정)

특징 순서는 학습 스크립트의 ``feature_cols``와 같다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

FEATURE_COLS = [
    "can_id", "dlc",
    "data_0", "data_1", "data_2", "data_3",
    "data_4", "data_5", "data_6", "data_7",
    "delta_t", "delta_t_id",
]
N_FEATURES = len(FEATURE_COLS)

# 학습 로그(train_*.log)에 기록된 99 분위수 클리핑 상한
DELTA_T_CLIP = 5.598068e-03
DELTA_T_ID_CLIP = 8.638672e+00


@dataclass
class CanFrame:
    stamp: float          # 초 단위 수신 시각
    can_id: int
    dlc: int
    data: bytes           # 길이 0~8
    label: int = 0        # 평가용 정답 (T=1 / R=0). 실시간 경로에서는 알 수 없음


def frame_bytes(dlc: int, data: Sequence[int]) -> List[int]:
    """DLC 이후 바이트를 0으로 만든 8바이트 리스트."""
    out = [0] * 8
    for i in range(min(dlc, 8, len(data))):
        out[i] = int(data[i])
    return out


class StreamingFeatureExtractor:
    """상태를 가진 온라인 특징 추출기.

    ``delta_t_id``를 위해 ID별 마지막 수신 시각을 들고 있다.
    같은 순서의 입력이면 :func:`compute_features_offline`과 같은 값을 낸다.
    """

    def __init__(self, clip: bool = True):
        self.clip = clip
        self._last_t: Optional[float] = None
        self._last_t_id: Dict[int, float] = {}

    def reset(self) -> None:
        self._last_t = None
        self._last_t_id.clear()

    def update(self, stamp: float, can_id: int, dlc: int, data: Sequence[int]) -> np.ndarray:
        dt = 0.0 if self._last_t is None else stamp - self._last_t
        prev = self._last_t_id.get(can_id)
        dt_id = 0.0 if prev is None else stamp - prev
        self._last_t = stamp
        self._last_t_id[can_id] = stamp
        if self.clip:
            dt = min(dt, DELTA_T_CLIP)
            dt_id = min(dt_id, DELTA_T_ID_CLIP)
        return np.array([can_id, dlc, *frame_bytes(dlc, data), dt, dt_id], dtype=np.float64)

    def update_frame(self, f: CanFrame) -> np.ndarray:
        return self.update(f.stamp, f.can_id, f.dlc, f.data)

    def transform(self, frames: Iterable[CanFrame]) -> np.ndarray:
        rows = [self.update_frame(f) for f in frames]
        return np.vstack(rows) if rows else np.empty((0, N_FEATURES))


def compute_features_offline(df, clip: bool = True) -> np.ndarray:
    """원 저장소 ``add_time_features`` + 학습 스크립트 클리핑의 배치(pandas) 구현.

    ``df``: timestamp, can_id(int), dlc, data_0..7(int, DLC 이후 0) 컬럼.
    입력 행 순서를 시간순으로 가정한다 (원본 로그는 이미 정렬돼 있음).
    """
    t = df["timestamp"]
    delta_t = t.diff().fillna(0.0)
    delta_t_id = t.groupby(df["can_id"]).diff().fillna(0.0)
    if clip:
        delta_t = delta_t.clip(upper=DELTA_T_CLIP)
        delta_t_id = delta_t_id.clip(upper=DELTA_T_ID_CLIP)
    X = np.empty((len(df), N_FEATURES), dtype=np.float64)
    X[:, 0] = df["can_id"].to_numpy()
    X[:, 1] = df["dlc"].to_numpy()
    for i in range(8):
        X[:, 2 + i] = df[f"data_{i}"].to_numpy()
    X[:, 10] = delta_t.to_numpy()
    X[:, 11] = delta_t_id.to_numpy()
    return X
