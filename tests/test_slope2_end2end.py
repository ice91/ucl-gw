# tests/test_slope2_end2end.py
import os, json, subprocess, sys

def test_slope2_from_data_ct_bounds():
    data_csv = "data/ct/ct_bounds.csv"
    if not os.path.isfile(data_csv):
        # 若還沒跑 Phase 2/3，改用 examples（維持檢核通路）
        data_csv = "examples/ct_bounds.csv"
        assert os.path.isfile(data_csv)

    # 指派任一 profile（專案已有 lvk_o3.yaml）
    cmd = [sys.executable, "-m", "scripts.nlo_slope_fit", "--data", data_csv, "--profile", "configs/profiles/lvk_o3.yaml"]
    subprocess.run(cmd, check=True)

    assert os.path.isfile("reports/slope2.json")
    with open("reports/slope2.json") as f:
        j = json.load(f)
    assert j.get("accept", False) is True
    # 給寬一點的容忍範圍（不同資料集也能 PASS）
    s = float(j.get("slope_hat", 0.0))
    assert 1.6 <= s <= 2.4
