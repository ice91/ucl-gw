# tests/test_aggregate.py
import os, pandas as pd
from src.uclgw.combine.aggregate import aggregate

def test_aggregate_runs(tmp_path):
    # 使用專案實際資料夾
    events_dir = "data/ct/events"
    out_csv = "data/ct/ct_bounds.csv"

    # 若事件資料不存在，略過（讓 CI 在 Phase 2 未跑時不爆）
    if not os.path.isdir(events_dir) or not any(fn.endswith("_ct_bounds.csv") for fn in os.listdir(events_dir)):
        assert True
        return

    summary = aggregate(events_dir=events_dir, out_csv=out_csv)
    assert os.path.isfile(out_csv)
    df = pd.read_csv(out_csv)
    # 基本欄位與至少一列
    for col in ["event", "ifo", "f_hz", "k", "delta_ct2", "sigma"]:
        assert col in df.columns
    assert len(df) >= 1
    # 單調排序檢查（事件/IFO/頻率）
    df2 = df.sort_values(by=["event","ifo","f_hz"]).reset_index(drop=True)
    assert (df2["f_hz"].values == df["f_hz"].values).all()
