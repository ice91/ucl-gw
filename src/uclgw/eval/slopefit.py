# src/uclgw/eval/slopefit.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
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

def _mask_profile(df: pd.DataFrame, profile_yaml: Optional[Union[str, Path]]):
    if not profile_yaml:
        return df
    prof = yaml.safe_load(Path(profile_yaml).read_text())
    fmin = prof.get("fmin", None)
    fmax = prof.get("fmax", None)
    out = df.copy()
    if fmin is not None:
        out = out[out["f_hz"] >= float(fmin)]
    if fmax is not None:
        out = out[out["f_hz"] <= float(fmax)]
    return out

def _mad_scale(resid: np.ndarray) -> float:
    # median absolute deviation → 正態一致性常數 1.4826
    mad = np.median(np.abs(resid - np.median(resid)))
    s = 1.4826 * mad
    if not np.isfinite(s) or s < 1e-12:
        s = np.std(resid, ddof=1)
    return float(max(s, 1e-12))

def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    W = np.diag(w)
    XtW = X.T @ W
    beta = np.linalg.inv(XtW @ X) @ (XtW @ y)
    return beta

def _irls_huber(
    X: np.ndarray,
    y: np.ndarray,
    w_meas: np.ndarray,
    c: float = 1.345,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Huber IRLS with leverage-standardized residuals:
    r_std = r / (s * sqrt(1 - h_ii)), where h_ii is the weighted hat diagonal.
    """
    # 初值：WLS
    beta = _wls(X, y, w_meas)

    for _ in range(max_iter):
        yhat = X @ beta
        r = y - yhat

        # robust 尺度
        s = _mad_scale(r)

        # 以當前的有效權重估帽子矩陣對角（加權版）
        # W^{1/2}X (X^T W X)^{-1} X^T W^{1/2}
        Wsqrt = np.sqrt(w_meas)
        Z = (Wsqrt[:, None] * X)                 # Z = W^{1/2} X
        XtWX = Z.T @ Z
        XtWX_inv = np.linalg.inv(XtWX)
        # h_ii = row-wise dot of Z @ XtWX_inv with Z
        H = Z @ XtWX_inv @ Z.T
        h = np.clip(np.diag(H), 0.0, 0.999999)   # 穩定化

        # 槓桿標準化殘差
        r_std = r / (s * np.sqrt(1.0 - h))

        # Huber 權重（psi(u)/u）
        w_psi = np.ones_like(r_std)
        big = np.abs(r_std) > c
        w_psi[big] = c / np.abs(r_std[big])

        # 疊乘 measurement-weights → 有效權重
        w_eff = w_meas * w_psi

        beta_new = _wls(X, y, w_eff)
        if np.linalg.norm(beta_new - beta) < tol * (1.0 + np.linalg.norm(beta)):
            beta = beta_new
            break
        beta = beta_new

    return beta

def do_fit(
    data_csv: Path,
    profile_yaml: Optional[Union[str, Path]],
    method: str = "wls"
) -> FitResult:
    df = load_ct(data_csv)
    df = _mask_profile(df, profile_yaml)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["delta_ct2"] > 0]

    x = np.log10(df["k"].values.astype(float))
    y = np.log10(df["delta_ct2"].values.astype(float))
    # measurement-weights：若無 sigma 欄位，給一個溫和的固定誤差
    sigma = df.get("sigma", pd.Series(np.full(len(df), 1e-2))).values.astype(float)
    w_meas = 1.0 / (np.maximum(sigma, 1e-12) ** 2)

    X = np.vstack([x, np.ones_like(x)]).T

    if method == "wls":
        beta = _wls(X, y, w_meas)
        yhat = X @ beta
        return FitResult(float(beta[0]), float(beta[1]), x, y, w_meas, yhat)

    elif method == "huber":
        beta = _irls_huber(X, y, w_meas, c=1.345)
        yhat = X @ beta
        return FitResult(float(beta[0]), float(beta[1]), x, y, w_meas, yhat)

    # 未知方法 → 退回 OLS（相容性）
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    return FitResult(float(beta[0]), float(beta[1]), x, y, w_meas, yhat)
