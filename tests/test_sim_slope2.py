# tests/test_sim_slope2.py
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from uclgw.sim.synth import SynthConfig, synth_ct_bounds
from uclgw.eval.slopefit import do_fit

def _fit_slope_on_df(df: pd.DataFrame, method: str = "huber") -> float:
    tmp = ROOT / "data/ct/_tmp_sim.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    #r = do_fit(tmp, profile_path=None, method=method)
    r = do_fit(tmp, None, method=method)
    return float(r.slope)

def test_estimator_accuracy_no_outlier():
    cfg = SynthConfig(event="SIM_SMOKE", sigma_rel=0.10, p_out=0.0, slope=2.0, n_bins=24, seed=1)
    df = synth_ct_bounds(cfg)
    s_hat_h = _fit_slope_on_df(df, "huber")
    s_hat_w = _fit_slope_on_df(df, "wls")
    # 允許 ±0.2 誤差帶（工程門檻）
    assert 1.8 <= s_hat_h <= 2.2
    assert 1.8 <= s_hat_w <= 2.2

def test_robustness_outliers_huber_better():
    cfg = SynthConfig(event="SIM_OUT", sigma_rel=0.15, p_out=0.05, out_mult=20.0, slope=2.0, n_bins=24, seed=7)
    df = synth_ct_bounds(cfg)
    s_w = _fit_slope_on_df(df, "wls")
    s_h = _fit_slope_on_df(df, "huber")
    # Huber 應比 WLS 更接近 2
    assert abs(s_h - 2.0) <= abs(s_w - 2.0) + 1e-9

def test_null_slope_near_zero():
    cfg = SynthConfig(event="SIM_NULL", sigma_rel=0.10, p_out=0.0, slope=0.0, n_bins=24, seed=3)
    df = synth_ct_bounds(cfg)
    s_h = _fit_slope_on_df(df, "huber")
    # 真 Null，斜率應不顯著偏離 0（寬鬆閾值）
    assert abs(s_h) < 0.5
