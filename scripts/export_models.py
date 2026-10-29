#!/usr/bin/env python3
"""원 저장소 모델(.pkl/.pth) → 배포용 형식으로 변환하고 결과가 같은지 검증한다.

- XGBoost          : xgb_model.pkl → xgb_model.json (네이티브 JSON)
- Isolation Forest : isolation_forest.pkl → isolation_forest.onnx (skl2onnx)
- LSTM Autoencoder : lstm_autoencoder.pth → lstm_autoencoder.ts (TorchScript) / .onnx
- 스케일러 3종     : *_scaler.pkl → model_meta.json 안의 mean/scale

pickle은 임의 코드를 실행할 수 있으므로 numpy/sklearn/xgboost 클래스만 허용하는
제한 언피클러로 읽는다. .pth는 ``weights_only=True``로 읽는다.

사용법:
    python scripts/export_models.py --upstream ../PROJECT-CAN-Anomaly-Detection/models \
        --sample data/Data/raw/DoS_dataset.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "can_ids"))

from can_ids.dataset import load_hcrl_csv  # noqa: E402
from can_ids.features import (DELTA_T_CLIP, DELTA_T_ID_CLIP, FEATURE_COLS,  # noqa: E402
                              compute_features_offline)

LSTM_SEQ_LEN = 50
LSTM_THRESHOLD = 115.286392  # train_lstm_autoencoder.log: Threshold (mean+3std)

_ALLOWED_PREFIX = ("numpy", "sklearn.", "xgboost.", "joblib.numpy_pickle")
_ALLOWED_EXACT = {("builtins", "bytearray"), ("builtins", "slice"), ("collections", "OrderedDict")}


def safe_joblib_load(path: Path):
    import joblib.numpy_pickle as jp

    class RestrictedUnpickler(jp.NumpyUnpickler):
        def find_class(self, module, name):
            if module.startswith(_ALLOWED_PREFIX) or (module, name) in _ALLOWED_EXACT:
                return super().find_class(module, name)
            raise RuntimeError(f"pickle에 허용되지 않은 전역 참조: {module}.{name}")

    orig = jp.NumpyUnpickler
    jp.NumpyUnpickler = RestrictedUnpickler
    try:
        return jp.load(path)
    finally:
        jp.NumpyUnpickler = orig


def lstm_module():
    import torch.nn as nn

    class LSTMAutoencoder(nn.Module):  # 원 저장소 train_lstm_autoencoder.py와 동일
        def __init__(self, input_dim=12, hidden_dim=32, latent_dim=16):
            super().__init__()
            self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.enc_out = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
            self.dec_out = nn.Linear(hidden_dim, input_dim)

        def forward(self, x):
            enc, _ = self.encoder(x)
            latent = self.enc_out(enc[:, -1, :])
            latent_rep = latent.unsqueeze(1).repeat(1, x.size(1), 1)
            dec, _ = self.decoder(latent_rep)
            return self.dec_out(dec)

    return LSTMAutoencoder()


def scaler_dict(s) -> dict:
    return {"mean": s.mean_.tolist(), "scale": s.scale_.tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", required=True, type=Path, help="원 저장소 models/ 디렉토리")
    ap.add_argument("--out", default=ROOT / "models", type=Path)
    ap.add_argument("--sample", type=Path, help="검증용 HCRL CSV (앞 20,000행 사용)")
    args = ap.parse_args()
    up, out = args.upstream, args.out
    out.mkdir(parents=True, exist_ok=True)

    import torch
    import xgboost as xgb
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as ort

    scalers = {k: safe_joblib_load(up / f"{f}_scaler.pkl")
               for k, f in [("xgb", "xgb"), ("iforest", "isolation_forest"), ("lstm", "lstm")]}
    xgb_clf = safe_joblib_load(up / "xgb_model.pkl")
    iforest = safe_joblib_load(up / "isolation_forest.pkl")
    iforest.set_params(n_jobs=1)

    # 검증 입력: 실제 로그 특징 (없으면 정규분포 난수)
    if args.sample:
        X = compute_features_offline(load_hcrl_csv(args.sample, nrows=20000))
    else:
        rng = np.random.default_rng(0)
        X = scalers["xgb"].inverse_transform(rng.normal(size=(5000, 12)))

    # ── XGBoost
    booster = xgb_clf.get_booster()
    booster.save_model(str(out / "xgb_model.json"))
    b2 = xgb.Booster()
    b2.load_model(str(out / "xgb_model.json"))
    Xs = scalers["xgb"].transform(X)
    p_ref = xgb_clf.predict_proba(Xs)[:, 1]
    p_new = b2.inplace_predict(Xs)
    print(f"[xgb]     max|Δp| = {np.abs(p_ref - p_new).max():.2e}, "
          f"판정 불일치 = {int(((p_ref > .5) != (p_new > .5)).sum())}/{len(X)}")

    # ── Isolation Forest → ONNX
    onx = convert_sklearn(iforest, initial_types=[("input", FloatTensorType([None, 12]))],
                          target_opset={"": 17, "ai.onnx.ml": 3})
    (out / "isolation_forest.onnx").write_bytes(onx.SerializeToString())
    sess = ort.InferenceSession(str(out / "isolation_forest.onnx"), providers=["CPUExecutionProvider"])
    outs = [o.name for o in sess.get_outputs()]
    Xi = scalers["iforest"].transform(X)
    d_ref = iforest.decision_function(Xi)
    res = dict(zip(outs, sess.run(None, {"input": Xi.astype(np.float32)})))
    score_output = "scores"
    d_new = res[score_output].reshape(-1)
    print(f"[iforest] outputs={outs}, max|Δdecision| = {np.abs(d_ref - d_new).max():.2e}, "
          f"판정 불일치 = {int(((d_ref < 0) != (d_new < 0)).sum())}/{len(X)}")

    # ── LSTM Autoencoder → TorchScript / ONNX
    model = lstm_module()
    model.load_state_dict(torch.load(up / "lstm_autoencoder.pth", map_location="cpu", weights_only=True))
    model.eval()
    ts = torch.jit.script(model)
    ts.save(str(out / "lstm_autoencoder.ts"))
    dummy = torch.zeros(1, LSTM_SEQ_LEN, 12)
    torch.onnx.export(model, dummy, str(out / "lstm_autoencoder.onnx"), input_names=["input"],
                      output_names=["recon"], dynamic_axes={"input": {0: "batch"}, "recon": {0: "batch"}},
                      opset_version=17, dynamo=False)
    Xl = scalers["lstm"].transform(X).astype(np.float32)
    w = np.lib.stride_tricks.sliding_window_view(Xl, LSTM_SEQ_LEN, axis=0).transpose(0, 2, 1)[:2000].copy()
    with torch.no_grad():
        r_ref = model(torch.from_numpy(w)).numpy()
        r_ts = ts(torch.from_numpy(w)).numpy()
    lsess = ort.InferenceSession(str(out / "lstm_autoencoder.onnx"), providers=["CPUExecutionProvider"])
    r_onnx = lsess.run(None, {"input": w})[0]
    e_ref = ((w - r_ref) ** 2).mean(axis=(1, 2))
    e_onnx = ((w - r_onnx) ** 2).mean(axis=(1, 2))
    print(f"[lstm]    max|Δrecon| ts={np.abs(r_ref - r_ts).max():.2e} onnx={np.abs(r_ref - r_onnx).max():.2e}, "
          f"max|Δerr|/err = {np.max(np.abs(e_ref - e_onnx) / np.maximum(e_ref, 1e-9)):.2e}")

    meta = {
        "source": "https://github.com/cbnu-yhjeon/PROJECT-CAN-Anomaly-Detection (models/)",
        "feature_cols": FEATURE_COLS,
        "clip": {"delta_t": DELTA_T_CLIP, "delta_t_id": DELTA_T_ID_CLIP},
        "scalers": {k: scaler_dict(s) for k, s in scalers.items()},
        "xgb": {"file": "xgb_model.json", "threshold": 0.5},
        "iforest": {"onnx": "isolation_forest.onnx", "score_output": score_output,
                    "offset": float(iforest.offset_), "contamination": iforest.contamination},
        "lstm": {"torchscript": "lstm_autoencoder.ts", "onnx": "lstm_autoencoder.onnx",
                 "seq_len": LSTM_SEQ_LEN, "threshold": LSTM_THRESHOLD},
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
