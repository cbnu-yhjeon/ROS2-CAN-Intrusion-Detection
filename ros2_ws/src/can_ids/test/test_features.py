import numpy as np
import pandas as pd

from can_ids.dataset import iter_frames, load_hcrl_csv
from can_ids.features import (DELTA_T_CLIP, DELTA_T_ID_CLIP, StreamingFeatureExtractor,
                              compute_features_offline)
from conftest import segment_or_skip

RAW = """1.000000,0316,8,05,21,68,09,21,21,00,6f,R
1.000200,018f,8,fe,5b,00,00,00,3c,00,00,R
1.000300,02b0,5,ff,7f,00,05,49,R
1.000400,0000,8,00,00,00,00,00,00,00,00,T
1.000700,0316,8,05,21,68,09,21,21,00,70,R
20.000000,0316,2,01,02,R
"""


def raw_df(tmp_path):
    p = tmp_path / "raw.csv"
    p.write_text(RAW)
    return load_hcrl_csv(p)


def test_loader_reads_flag_after_short_dlc(tmp_path):
    df = raw_df(tmp_path)
    assert df["label"].tolist() == [0, 0, 0, 1, 0, 0]
    assert df.loc[2, ["data_5", "data_6", "data_7"]].tolist() == [0, 0, 0]
    assert df.loc[2, "data_4"] == 0x49
    assert df.loc[5, "dlc"] == 2 and df.loc[5, "data_2"] == 0


def test_streaming_matches_offline_on_synthetic(tmp_path):
    df = raw_df(tmp_path)
    off = compute_features_offline(df)
    on = StreamingFeatureExtractor().transform(iter_frames(df))
    np.testing.assert_array_equal(off, on)


def test_time_features_and_clipping(tmp_path):
    X = compute_features_offline(raw_df(tmp_path))
    assert X[0, 10] == 0 and X[0, 11] == 0               # 첫 메시지
    np.testing.assert_allclose(X[4, 11], 0.0007, atol=1e-9)  # 같은 ID 0x316 간격
    assert X[5, 10] == DELTA_T_CLIP and X[5, 11] == DELTA_T_ID_CLIP


def test_matches_upstream_pandas_pipeline(tmp_path):
    """원 저장소 add_time_features + 학습 스크립트 클리핑을 그대로 재현한 것과 비교."""
    df = raw_df(tmp_path)
    up = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    up["delta_t"] = up["timestamp"].diff().fillna(0).clip(upper=DELTA_T_CLIP)
    up["delta_t_id"] = up.groupby("can_id")["timestamp"].diff().fillna(0).clip(upper=DELTA_T_ID_CLIP)
    cols = ["can_id", "dlc"] + [f"data_{i}" for i in range(8)] + ["delta_t", "delta_t_id"]
    np.testing.assert_array_equal(up[cols].to_numpy(float), compute_features_offline(df))


def test_streaming_matches_offline_on_real_segment():
    df = segment_or_skip("fuzzy", 30000)
    off = compute_features_offline(df)
    on = StreamingFeatureExtractor().transform(iter_frames(df))
    np.testing.assert_array_equal(off, on)
