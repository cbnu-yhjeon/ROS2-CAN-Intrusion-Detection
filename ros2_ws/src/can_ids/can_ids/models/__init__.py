"""탐지기 구현. 모두 ``score(X) -> (scores, preds)`` 인터페이스를 따른다.

X는 :data:`can_ids.features.FEATURE_COLS` 순서의 (n, 12) 원시 특징이고,
스케일링은 각 탐지기 안에서 한다 (모델마다 학습한 스케일러가 다름).
"""
from __future__ import annotations

import json
from pathlib import Path

from can_ids.models.base import Detector, StandardScaler

MODEL_NAMES = ("xgb", "iforest", "lstm", "ensemble")


def load_meta(model_dir) -> dict:
    return json.loads((Path(model_dir) / "model_meta.json").read_text())


def load_detector(name: str, model_dir, backend: str = "onnx", threads: int = 1, **kw) -> Detector:
    model_dir = Path(model_dir)
    meta = load_meta(model_dir)
    if name == "xgb":
        from can_ids.models.xgb import XGBDetector
        return XGBDetector(model_dir, meta, threads=threads, **kw)
    if name == "iforest":
        from can_ids.models.iforest import IForestDetector
        return IForestDetector(model_dir, meta, threads=threads, **kw)
    if name == "lstm":
        from can_ids.models.lstm_ae import LSTMAEDetector
        return LSTMAEDetector(model_dir, meta, backend=backend, threads=threads, **kw)
    if name == "ensemble":
        from can_ids.models.ensemble import EnsembleDetector
        from can_ids.models.lstm_ae import LSTMAEDetector
        from can_ids.models.xgb import XGBDetector
        return EnsembleDetector(XGBDetector(model_dir, meta, threads=threads),
                                LSTMAEDetector(model_dir, meta, backend=backend, threads=threads))
    raise ValueError(f"unknown model {name!r}, expected one of {MODEL_NAMES}")


__all__ = ["Detector", "StandardScaler", "MODEL_NAMES", "load_detector", "load_meta"]
