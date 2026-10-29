import numpy as np
import pytest

from can_ids.features import compute_features_offline
from can_ids.models import load_detector
from conftest import MODEL_DIR, segment_or_skip


@pytest.fixture(scope="module")
def X():
    return compute_features_offline(segment_or_skip("dos", 3000))


@pytest.mark.parametrize("name", ["xgb", "iforest", "lstm", "ensemble"])
def test_per_message_equals_batch(name, X):
    det = load_detector(name, MODEL_DIR)
    s_batch, p_batch = det.score(X)
    det.reset()
    per = [det.score(X[i:i + 1]) for i in range(len(X))]
    s_one = np.concatenate([s for s, _ in per])
    p_one = np.concatenate([p for _, p in per])
    if name != "ensemble":  # 앙상블 점수는 XGB 확률이라 동일, 판정만 비교
        np.testing.assert_allclose(s_one, s_batch, rtol=1e-4, atol=1e-6)
    assert (p_one == p_batch).mean() > 0.999


def test_lstm_windows_match_training_sequences(X):
    """원 저장소 SequenceDataset: 시퀀스 i = data[i:i+L], 라벨 = 마지막 행."""
    det = load_detector("lstm", MODEL_DIR)
    L = det.seq_len
    s, _ = det.score(X)
    assert (s[:L - 1] == 0).all()
    Xs = det.scaler.transform(X).astype(np.float32)
    w = np.stack([Xs[i:i + L] for i in range(len(X) - L + 1)])
    np.testing.assert_allclose(s[L - 1:], det.recon_error(w), rtol=1e-5)


def test_ensemble_is_subset_of_xgb(X):
    xgb = load_detector("xgb", MODEL_DIR)
    ens = load_detector("ensemble", MODEL_DIR)
    _, px = xgb.score(X)
    _, pe = ens.score(X)
    assert not (pe & ~px).any()


def test_xgb_detects_dos_injection(X):
    seg = segment_or_skip("dos", 3000)
    _, p = load_detector("xgb", MODEL_DIR).score(X)
    y = seg["label"].to_numpy().astype(bool)
    assert y.any() and (p == y).mean() > 0.99
