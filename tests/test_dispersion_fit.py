# tests/test_dispersion_fit.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from uclgw.io.gwosc import synthesize_event
from uclgw.preprocess.conditioning import prepare_event
from uclgw.features.dispersion_fit import build_ct_points_from_whitened

def _slope_loglog(x, y):
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1])

def test_proxy_k2_slope(tmp_path):
    # 合成 & 白化
    raw = tmp_path / "raw/gwosc"
    seg = tmp_path / "work/segments"
    synthesize_event("GW170817", ["H1"], fs=2048.0, duration=16.0, seed=11,
                     raw_root=raw, seg_root=seg)
    segj = json.loads((seg/"GW170817.json").read_text())
    res = prepare_event(segj, psd_dir=tmp_path/"work/psd", white_dir=tmp_path/"work/whitened",
                        nperseg=2048, noverlap=1024, gate_z=5.0)
    df = build_ct_points_from_whitened([Path(res["whitened"][0])], fmin=30, fmax=512,
                                       n_bins=16, mode="proxy-k2")
    # 斜率 ~ 2（在 log10 空間檢查）
    x = np.log10(df["k"].values + 1e-40)
    y = np.log10(df["delta_ct2"].values + 1e-40)
    s = _slope_loglog(x, y)
    assert 1.8 <= s <= 2.2
