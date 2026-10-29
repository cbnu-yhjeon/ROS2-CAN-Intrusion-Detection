from __future__ import annotations

from typing import Tuple

import numpy as np


class StandardScaler:
    """scikit-learn StandardScaler와 같은 변환. 파라미터는 JSON에서 읽는다."""

    def __init__(self, mean, scale):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)

    @classmethod
    def from_meta(cls, d: dict) -> "StandardScaler":
        return cls(d["mean"], d["scale"])

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale


class Detector:
    """메시지 단위 탐지기 인터페이스."""

    name = "base"
    threshold = 0.0

    def score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """X: (n, 12) 원시 특징 → (점수, 공격 여부 bool)."""
        raise NotImplementedError

    def reset(self) -> None:
        """스트림 상태 초기화 (상태가 있는 모델만)."""
