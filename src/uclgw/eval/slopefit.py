# src/uclgw/eval/slopefit.py
from __future__ import annotations
import json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import yaml

@dataclass
class FitResult:
    slope: float
    intercept: float
    n: int
    method: str
    mask_count: int
    window_k: Tuple[float, float]

def _read_profile_window(profile_yaml: Optional[Path]) -> Tuple[float, float]:
    if profile_yaml is None:
        return (None, None)
    d = yaml.safe_load(Path(profile_yaml).read_text())
    # 允許兩種寫法：直接給 k window；或給 f window 後轉 k
    if "window_k" in d:
        return tuple(d["window_k"])
    if "window_f_hz" in d:
        fmin, fmax = d["window_f_hz"]
        # k ~ 2π f / c，常數對斜率影響為 0；這裡僅作窗口裁剪
        c = 299792458.0
        return (2*np.pi*fmin/c, 2*np.pi*fmax/c)
    return (None, None)

def load_ct(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 合法值過濾
    df = df[(df["delta_ct2"] > 0) & (df["sigma"] > 0)]
    # 轉成 log 空間（加極小值避免除零）
    eps = 1e-300
    df["x"] = np.log10(df["k"].values + eps)
    df["y"] = np.log10(df["delta_ct2"].values + eps)
    # σ_y 由誤差傳播：σ_y = σ / (y * ln10)；此處 y→delta_ct2
    df["sigma_y"] = df["sigma"] / ((df["delta_ct2"] + eps) * math.log(10))
    # 權重
    df["w"] = 1.0 / (df["sigma_y"]**2 + 1e-30)
    return df

def _weighted_ls(x, y, w):
    W = np.diag(w)
    X = np.stack([x, np.ones_like(x)], axis=1)
    XtW = X.T @ W
    beta = np.linalg.pinv(XtW @ X) @ (XtW @ y)
    slope, intercept = beta[0], beta[1]
    return slope, intercept

def _huber_weights(resid, c=1.5):
    # Huber：|r|<=c → 1；否則 c/|r|
    r = np.abs(resid)
    w = np.ones_like(r)
    bad = r > c
    w[bad] = (c / (r[bad] + 1e-12))
    return w

def fit_line(df: pd.DataFrame, method: str = "wls", huber_c: float = 1.5):
    x = df["x"].values
    y = df["y"].values
    w = df["w"].values
    if method == "wls":
        return _weighted_ls(x, y, w)
    elif method == "huber":
        # IRLS with Huber
        slope, intercept = _weighted_ls(x, y, w)
        for _ in range(10):
            yhat = slope * x + intercept
            r = (y - yhat) * np.sqrt(w)
            hw = _huber_weights(r, c=huber_c)
            w_new = w * hw
            slope, intercept = _weighted_ls(x, y, w_new)
        return slope, intercept
    else:
        raise ValueError(f"unknown method={method}")

def window_mask(df: pd.DataFrame, kmin: Optional[float], kmax: Optional[float]) -> np.ndarray:
    if kmin is None and kmax is None:
        return np.ones(len(df), dtype=bool)
    k = df["k"].values
    m = np.ones_like(k, dtype=bool)
    if kmin is not None: m &= (k >= kmin)
    if kmax is not None: m &= (k <= kmax)
    return m

def do_fit(csv_path: Path, profile_yaml: Optional[Path]=None, method: str="wls",
           ifo: Optional[str]=None, event: Optional[str]=None,
           k_window_override: Optional[Tuple[float,float]] = None) -> FitResult:
    df = load_ct(csv_path)
    if ifo is not None:
        df = df[df["ifo"] == ifo]
    if event is not None:
        df = df[df["event"] == event]
    kmin, kmax = _read_profile_window(profile_yaml) if k_window_override is None else k_window_override
    m = window_mask(df, kmin, kmax)
    df2 = df[m].copy()
    slope, intercept = fit_line(df2, method=method)
    return FitResult(slope=slope, intercept=intercept, n=len(df2),
                     method=method, mask_count=int(m.sum()),
                     window_k=(kmin if kmin else float(df2["k"].min()),
                               kmax if kmax else float(df2["k"].max())))
