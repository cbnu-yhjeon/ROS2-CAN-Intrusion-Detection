import numpy as np
import pytest

from can_ids.dataset import iter_frames
from can_ids.engine import DetectionEngine, run_stream
from can_ids.evaluate import classification, offline_predictions, streaming_predictions
from can_ids.models import load_detector
from conftest import MODEL_DIR, segment_or_skip


@pytest.mark.parametrize("batch", [1, 7, 64])
def test_batch_size_does_not_change_verdicts(batch):
    seg = segment_or_skip("gear", 4000)
    det = load_detector("lstm", MODEL_DIR)
    ref = run_stream(DetectionEngine(det), iter_frames(seg), 1)
    det.reset()
    got = run_stream(DetectionEngine(det), iter_frames(seg), batch)
    assert [v.is_attack for v in ref] == [v.is_attack for v in got]
    assert [v.index for v in got] == list(range(len(seg)))
    assert max(v.batch_size for v in got) == batch


@pytest.mark.parametrize("name", ["xgb", "iforest", "lstm"])
def test_offline_online_f1_gap_under_1pp(name):
    """G3: 같은 로그에서 오프라인 배치 경로와 스트리밍 경로의 F1 차이 1%p 이내."""
    seg = segment_or_skip("rpm", 20000)
    det = load_detector(name, MODEL_DIR)
    y = seg["label"].to_numpy()
    f_off = classification(y, offline_predictions(seg, det))["f1"]
    f_on = classification(y, streaming_predictions(seg, det, batch_size=16))["f1"]
    assert abs(f_off - f_on) * 100 < 1.0
