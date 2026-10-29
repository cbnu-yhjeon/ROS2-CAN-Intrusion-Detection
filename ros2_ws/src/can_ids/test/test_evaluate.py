import numpy as np
import pandas as pd

from can_ids.evaluate import align, bursts, classification, detection_delays


def seg_df():
    rows = [(0.0, 1, [1]), (0.1, 2, [2]), (0.2, 1, [1]), (0.3, 3, [3]), (0.4, 1, [9])]
    df = pd.DataFrame({"timestamp": [r[0] for r in rows], "can_id": [r[1] for r in rows],
                       "dlc": [1] * len(rows)})
    for i in range(8):
        df[f"data_{i}"] = [r[2][0] if i == 0 else 0 for r in rows]
    return df


def test_align_skips_lost_frames():
    seg = seg_df()
    v = pd.DataFrame({"can_id": [1, 1, 3, 1], "data": ["01", "01", "03", "09"]})
    # 두 번째 프레임(id 2) 유실
    assert align(seg, v).tolist() == [0, 2, 3, 4]


def test_classification():
    m = classification(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["f1"] == 0.5


def test_bursts_and_delays():
    stamps = np.array([0.0, 0.1, 0.2, 0.3, 2.0, 2.1, 5.0])
    labels = np.array([0, 1, 1, 0, 1, 1, 1])
    assert bursts(labels, stamps, gap_s=0.5) == [(1, 2), (4, 5), (6, 6)]
    preds = np.array([0, 0, 1, 0, 0, 0, 0])
    t_done = stamps + 0.001
    d = detection_delays(labels, stamps, preds, t_done)
    assert d["bursts"] == 3 and d["missed_bursts"] == 2
    np.testing.assert_allclose(d["delay_p50_ms"], 101.0)
