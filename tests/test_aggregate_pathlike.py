# tests/test_aggregate_pathlike.py
from pathlib import Path
import pandas as pd
from src.uclgw.combine.aggregate import aggregate  # 與現有 test_aggregate.py 保持一致

def test_aggregate_accepts_pathlike(tmp_path: Path):
    events = tmp_path / "data/ct/events"
    out_csv = tmp_path / "data/ct/ct_bounds.csv"
    summary_json = tmp_path / "reports/summary.json"
    events.mkdir(parents=True, exist_ok=True)
    (summary_json.parent).mkdir(parents=True, exist_ok=True)

    # 準備一個極簡事件表
    df = pd.DataFrame({
        "event": ["E"],
        "ifo":   ["H1"],
        "f_hz":  [100.0],
        "k":     [1e-6],
        "delta_ct2": [1e-9],
        "sigma": [1e-10],
    })
    (events / "E_ct_bounds.csv").write_text(df.to_csv(index=False))

    summary = aggregate(events_dir=events, out_csv=out_csv, write_summary_json=summary_json)
    assert Path(summary.out_csv).exists()
    assert summary.n_rows == 1
    assert (summary_json).exists()

    # 讀回檢查
    out_df = pd.read_csv(out_csv)
    assert list(out_df.columns) == ["event","ifo","f_hz","k","delta_ct2","sigma"]
    assert len(out_df) == 1
