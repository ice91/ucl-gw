# tests/test_aggregate.py
from pathlib import Path
import pandas as pd
from uclgw.combine.aggregate import append_event_to_master

def test_append_and_dedup(tmp_path):
    master = tmp_path/"ct_bounds.csv"
    e1 = tmp_path/"ev1.csv"
    e2 = tmp_path/"ev2.csv"
    cols = ["event","ifo","f_hz","k","delta_ct2","sigma"]
    pd.DataFrame([
        ["E","H1", 100.0, 1.0, 1e-20, 1e-21],
        ["E","H1", 200.0, 2.0, 4e-20, 2e-21],
    ], columns=cols).to_csv(e1, index=False)
    pd.DataFrame([
        ["E","H1", 200.0, 2.0, 4e-20, 2e-21],  # 重複
        ["E","L1", 300.0, 3.0, 9e-20, 3e-21],
    ], columns=cols).to_csv(e2, index=False)

    append_event_to_master(e1, master)
    append_event_to_master(e2, master)
    m = pd.read_csv(master)
    # 去重後應 3 列
    assert len(m) == 3
    assert set(m["ifo"]) == {"H1","L1"}
