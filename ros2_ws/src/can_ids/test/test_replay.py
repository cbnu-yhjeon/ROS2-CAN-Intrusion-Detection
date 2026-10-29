import numpy as np

from can_ids.dataset import load_hcrl_csv
from csv2candump import to_candump_lines
from pycanplayer import parse_candump


def test_candump_roundtrip(tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_text("1.000001,0316,8,05,21,68,09,21,21,00,6f,R\n1.000301,02b0,5,ff,7f,00,05,49,R\n")
    df = load_hcrl_csv(raw)
    lines = list(to_candump_lines(df))
    assert lines == ["(1.000001) vcan0 316#052168092121006F\n",
                     "(1.000301) vcan0 2B0#FF7F000549\n"]
    log = tmp_path / "x.log"
    log.write_text("".join(lines))
    stamps, frames = parse_candump(log)
    np.testing.assert_allclose(np.diff(stamps), [0.0003], atol=1e-9)
    assert frames[1][4] == 5 and frames[1][8:13] == bytes.fromhex("ff7f000549")
