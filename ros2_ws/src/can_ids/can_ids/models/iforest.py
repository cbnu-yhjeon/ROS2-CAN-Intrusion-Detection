from __future__ import annotations

import numpy as np
import onnxruntime as ort

from can_ids.models.base import Detector, StandardScaler


def make_session(path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


class IForestDetector(Detector):
    """ONNX로 내보낸 Isolation Forest.

    점수 = scikit-learn ``decision_function`` 부호 반전 (클수록 이상).
    ``decision_function < 0`` ⇔ ``predict == -1`` 이므로 임계값은 0.
    """

    name = "iforest"

    def __init__(self, model_dir, meta: dict, threads: int = 1, threshold: float = 0.0):
        m = meta["iforest"]
        self.sess = make_session(model_dir / m["onnx"], threads)
        self.input_name = self.sess.get_inputs()[0].name
        self.score_output = m["score_output"]
        self.scaler = StandardScaler.from_meta(meta["scalers"]["iforest"])
        self.threshold = threshold

    def score(self, X):
        Xs = self.scaler.transform(X).astype(np.float32)
        decision = self.sess.run([self.score_output], {self.input_name: Xs})[0].reshape(-1)
        s = -decision.astype(np.float64)
        return s, s > self.threshold
