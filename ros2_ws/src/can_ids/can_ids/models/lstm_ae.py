from __future__ import annotations

from typing import Optional

import numpy as np

from can_ids.models.base import Detector, StandardScaler


class LSTMAEDetector(Detector):
    """슬라이딩 윈도우 LSTM Autoencoder.

    학습 때와 같이 메시지 i의 점수는 윈도우 ``[i-L+1, i]``의 재구성 MSE다.
    윈도우가 다 차기 전(스트림의 첫 L-1개)은 점수 0, 정상으로 본다.
    ``score``에 넘기는 X는 직전 호출에 이어지는 연속 메시지여야 한다.
    """

    name = "lstm"

    def __init__(self, model_dir, meta: dict, backend: str = "onnx", threads: int = 1,
                 threshold: Optional[float] = None, seq_len: Optional[int] = None):
        m = meta["lstm"]
        self.seq_len = int(seq_len or m["seq_len"])
        self.threshold = m["threshold"] if threshold is None else threshold
        self.scaler = StandardScaler.from_meta(meta["scalers"]["lstm"])
        self.backend = backend
        if backend == "onnx":
            from can_ids.models.iforest import make_session
            self.sess = make_session(model_dir / m["onnx"], threads)
            self.input_name = self.sess.get_inputs()[0].name
        elif backend == "torchscript":
            import torch
            torch.set_num_threads(threads)
            self.torch = torch
            self.module = torch.jit.load(str(model_dir / m["torchscript"]), map_location="cpu").eval()
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.reset()

    def reset(self):
        self._hist = np.empty((0, len(self.scaler.mean)), dtype=np.float32)

    def push(self, X: np.ndarray) -> np.ndarray:
        """X를 스케일해 버퍼에 넣고, 직전 L-1개 이력과 X를 이어 붙인 배열을 돌려준다."""
        Xs = self.scaler.transform(X).astype(np.float32)
        buf = np.concatenate([self._hist, Xs], axis=0)
        self._hist = buf[-(self.seq_len - 1):] if self.seq_len > 1 else buf[:0]
        return buf

    def windows(self, buf: np.ndarray, n_new: int):
        """buf 끝 n_new 행 각각으로 끝나는 윈도우와, 윈도우가 완전한 행의 mask."""
        L = self.seq_len
        n_full = max(0, min(n_new, len(buf) - L + 1))
        valid = np.zeros(n_new, dtype=bool)
        valid[n_new - n_full:] = True
        if n_full == 0:
            return np.empty((0, L, buf.shape[1]), dtype=np.float32), valid
        view = np.lib.stride_tricks.sliding_window_view(buf, L, axis=0)  # (N-L+1, F, L)
        w = np.ascontiguousarray(view[-n_full:].transpose(0, 2, 1))
        return w, valid

    def recon_error(self, w: np.ndarray) -> np.ndarray:
        if len(w) == 0:
            return np.empty(0)
        if self.backend == "onnx":
            out = self.sess.run(None, {self.input_name: w})[0]
        else:
            with self.torch.no_grad():
                out = self.module(self.torch.from_numpy(w)).numpy()
        return np.mean((w - out) ** 2, axis=(1, 2)).astype(np.float64)

    def score(self, X):
        buf = self.push(X)
        w, valid = self.windows(buf, len(X))
        s = np.zeros(len(X))
        s[valid] = self.recon_error(w)
        return s, s > self.threshold
