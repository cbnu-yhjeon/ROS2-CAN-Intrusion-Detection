from __future__ import annotations

import numpy as np
import xgboost as xgb

from can_ids.models.base import Detector, StandardScaler


class XGBDetector(Detector):
    """네이티브 JSON XGBoost 모델. 점수 = 공격 확률."""

    name = "xgb"

    def __init__(self, model_dir, meta: dict, threads: int = 1, threshold: float = None):
        m = meta["xgb"]
        self.booster = xgb.Booster()
        self.booster.load_model(str(model_dir / m["file"]))
        self.booster.set_param({"nthread": threads})
        self.scaler = StandardScaler.from_meta(meta["scalers"]["xgb"])
        self.threshold = m["threshold"] if threshold is None else threshold

    def score(self, X):
        p = self.booster.inplace_predict(self.scaler.transform(X))
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        return p, p > self.threshold
