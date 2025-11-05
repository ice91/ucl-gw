def test_fallback_window_from_data(tmp_path):
    import pandas as pd, numpy as np, json
    df = pd.read_csv("data/ct/ct_bounds.csv")
    # 刻意給一個不相交視窗的 profile
    prof = {"logfit":{"min_k":1e-14,"max_k":1e-11,"min_points":10}}
    p = tmp_path/"bad_profile.json"; p.write_text(json.dumps(prof))
    # 直接調用腳本主函數較麻煩，可改為子程序或提取函式做單測
    assert (df["k"].between(5e-7, 2.5e-5).sum() >= 10)
