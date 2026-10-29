"""ROS와 무관한 탐지 엔진: 특징 추출 → (마이크로 배치) → 추론.

ROS 노드와 오프라인 스트리밍 시뮬레이션이 같은 코드를 쓰게 해서
온라인/오프라인 차이가 전송 경로에서만 생기도록 한다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import numpy as np

from can_ids.features import CanFrame, StreamingFeatureExtractor
from can_ids.models.base import Detector


@dataclass
class Verdict:
    index: int        # 엔진이 받은 순번 (0부터)
    stamp: float      # 프레임 수신 시각 (특징 계산에 쓴 시각)
    can_id: int
    dlc: int
    data: bytes
    score: float
    is_attack: bool
    t_recv: float     # 엔진에 들어온 벽시계 시각
    t_done: float     # 판정이 끝난 벽시계 시각
    batch_size: int


class DetectionEngine:
    def __init__(self, detector: Detector, clock: Callable[[], float] = time.time):
        self.detector = detector
        self.extractor = StreamingFeatureExtractor()
        self.clock = clock
        self._frames: List[CanFrame] = []
        self._rows: List[np.ndarray] = []
        self._recv: List[float] = []
        self.count = 0

    def reset(self) -> None:
        self.extractor.reset()
        self.detector.reset()
        self._frames.clear()
        self._rows.clear()
        self._recv.clear()
        self.count = 0

    @property
    def pending(self) -> int:
        return len(self._frames)

    @property
    def oldest_pending(self) -> Optional[float]:
        return self._recv[0] if self._recv else None

    def submit(self, frame: CanFrame, t_recv: Optional[float] = None) -> None:
        """특징은 수신 순서대로 즉시 계산한다 (상태 갱신). 추론은 :meth:`process`에서."""
        self._rows.append(self.extractor.update_frame(frame))
        self._frames.append(frame)
        self._recv.append(self.clock() if t_recv is None else t_recv)

    def process(self) -> List[Verdict]:
        if not self._frames:
            return []
        X = np.vstack(self._rows)
        scores, preds = self.detector.score(X)
        t_done = self.clock()
        n = len(self._frames)
        out = [
            Verdict(self.count + i, f.stamp, f.can_id, f.dlc, f.data, float(s), bool(p), r, t_done, n)
            for i, (f, s, p, r) in enumerate(zip(self._frames, scores, preds, self._recv))
        ]
        self.count += n
        self._frames.clear()
        self._rows.clear()
        self._recv.clear()
        return out


def run_stream(engine: DetectionEngine, frames: Iterable[CanFrame], batch_size: int = 1) -> List[Verdict]:
    """오프라인 스트리밍 시뮬레이션: 프레임을 순서대로 넣고 batch_size마다 추론."""
    out: List[Verdict] = []
    for f in frames:
        engine.submit(f)
        if engine.pending >= batch_size:
            out.extend(engine.process())
    out.extend(engine.process())
    return out
