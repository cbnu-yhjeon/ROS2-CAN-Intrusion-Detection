from __future__ import annotations

import numpy as np

from can_ids.models.base import Detector
from can_ids.models.lstm_ae import LSTMAEDetector
from can_ids.models.xgb import XGBDetector


class EnsembleDetector(Detector):
    """계층형 앙상블: XGBoost 1차 판정 → LSTM-AE 2차 확인.

    LSTM 윈도우 버퍼는 모든 메시지로 갱신하지만, 재구성 추론은 XGBoost가
    공격이라 본 메시지에만 돌린다. 경보 = XGB 양성 AND LSTM 재구성 오차 > 임계값.
    점수는 XGBoost 확률을 그대로 쓴다.
    """

    name = "ensemble"

    def __init__(self, xgb_det: XGBDetector, lstm_det: LSTMAEDetector):
        self.xgb = xgb_det
        self.lstm = lstm_det
        self.threshold = xgb_det.threshold

    def reset(self):
        self.lstm.reset()

    def score(self, X):
        p, flag = self.xgb.score(X)
        buf = self.lstm.push(X)
        confirmed = np.zeros(len(X), dtype=bool)
        if flag.any():
            w, valid = self.lstm.windows(buf, len(X))
            # w는 valid 행에 순서대로 대응
            sel = flag[valid]
            if sel.any():
                err = self.lstm.recon_error(w[sel])
                idx = np.flatnonzero(valid)[sel]
                confirmed[idx] = err > self.lstm.threshold
        return p, confirmed
