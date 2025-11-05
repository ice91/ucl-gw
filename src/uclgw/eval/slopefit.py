# src/uclgw/eval/slopefit.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

@dataclass
class FitResult:
    slope: float
    intercept: float
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    yhat: np.ndarray

def load_ct(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def _mask_profile(df: pd.DataFrame, profile_yaml: str | None):
    if not profile_yaml:
        return df
    prof = yaml.safe_load(Path(profile_yaml).read_text())
    fmin = prof.get("fmin", None); fmax = prof.get("fmax", None)
    out = df.copy()
    if fmin is not None:
        out = out[out["f_hz"] >= float(fmin)]
    if fmax is not None:
        out = out[out["f_hz"] <= float(fmax)]
    return out

def do_fit(data_csv: Path, profile_yaml: str | None, method: str = "wls") -> FitResult:
    df = load_ct(data_csv)
    df = _mask_profile(df, profile_yaml)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["delta_ct2"] > 0]
    x = np.log10(df["k"].values.astype(float))
    y = np.log10(df["delta_ct2"].values.astype(float))
    sigma = df.get("sigma", pd.Series(np.full(len(df), 1e-2))).values.astype(float)
    w = 1.0 / (np.maximum(sigma, 1e-12) ** 2)

    X = np.vstack([x, np.ones_like(x)]).T
    if method == "wls":
        W = np.diag(w)
        XtW = X.T @ W
        beta = np.linalg.inv(XtW @ X) @ (XtW @ y)
    else:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    return FitResult(float(beta[0]), float(beta[1]), x, y, w, yhat)
