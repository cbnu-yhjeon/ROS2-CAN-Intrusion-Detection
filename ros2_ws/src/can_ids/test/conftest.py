import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parents[2]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(ROOT / "scripts"))

MODEL_DIR = ROOT / "models"
SEGMENT_DIR = ROOT / "data" / "segments"


def segment_or_skip(name: str, nrows: int = None):
    from can_ids.dataset import read_segment
    path = SEGMENT_DIR / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"{path} 없음 (scripts/csv2candump.py로 생성)")
    df = read_segment(path)
    return df.iloc[:nrows].reset_index(drop=True) if nrows else df
