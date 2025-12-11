# src/uclgw/eval/slopefit.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np
import pandas as pd
import yaml

_EPS = 1e-12

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

def _safe_inv(M: np.ndarray, ridge: float = 1e-12) -> np.ndarray:
    # 數值穩定的逆 (ridge)
    I = np.eye(M.shape[0], dtype=M.dtype)
    return np.linalg.inv(M + ridge * I)

def _mad_scale(resid: np.ndarray) -> float:
    mad = np.median(np.abs(resid - np.median(resid)))
    s = 1.4826 * mad
    if not np.isfinite(s) or s < _EPS:
        s = np.std(resid, ddof=1)
    return float(max(s, _EPS))

def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    Wsqrt = np.sqrt(np.clip(w, _EPS, None))[:, None]
    Z = Wsqrt * X                       # Z = W^{1/2} X
    XtWX = Z.T @ Z
    XtWX_inv = _safe_inv(XtWX)
    beta = XtWX_inv @ (Z.T @ (Wsqrt.flatten() * y))
    return beta

def _hat_diag(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    # 加權帽子矩陣對角：H = W^{1/2} X (X^T W X)^{-1} X^T W^{1/2}
    Wsqrt = np.sqrt(np.clip(w, _EPS, None))[:, None]
    Z = Wsqrt * X
    XtWX = Z.T @ Z
    XtWX_inv = _safe_inv(XtWX)
    H = Z @ XtWX_inv @ Z.T
    h = np.clip(np.diag(H), 0.0, 1.0 - 1e-9)
    return h

def _irls_huber(
    X: np.ndarray,
    y: np.ndarray,
    w_meas: np.ndarray,
    c: float = 1.345,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Huber IRLS with leverage-standardized residuals.
    每回合用「當前有效權重」計算帽子矩陣；收斂即停。
    回傳 (beta, w_eff)
    """
    # 初值：只用測量權重做 WLS
    w_eff = w_meas.copy()
    beta = _wls(X, y, w_eff)

    for _ in range(max_iter):
        yhat = X @ beta
        r = y - yhat
        s = _mad_scale(r)

        # 用「目前的 w_eff」算帽子對角（不是固定 w_meas）
        h = _hat_diag(X, w_eff)
        r_std = r / (s * np.sqrt(1.0 - h))

        w_psi = np.ones_like(r_std)
        big = np.abs(r_std) > c
        w_psi[big] = c / np.clip(np.abs(r_std[big]), _EPS, None)

        w_next = w_meas * w_psi
        beta_new = _wls(X, y, w_next)

        if np.linalg.norm(beta_new - beta) < tol * (1.0 + np.linalg.norm(beta)):
            beta = beta_new
            w_eff = w_next
            break

        beta, w_eff = beta_new, w_next

    return beta, w_eff

def _prepare_xyw(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 過濾、log 映射與誤差傳播
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["delta_ct2"] > 0]

    x = np.log10(df["k"].values.astype(float))
    y_lin = df["delta_ct2"].values.astype(float)
    y = np.log10(y_lin)

    sigma_y = df.get("sigma", pd.Series(np.full(len(df), 1e-2))).values.astype(float)
    ln10 = np.log(10.0)
    sigma_log = sigma_y / (np.maximum(y_lin, 1e-30) * ln10)
    w_meas = 1.0 / (np.maximum(sigma_log, _EPS) ** 2)

    return x, y, w_meas

def do_fit(
    data_csv: Path,
    profile_yaml: Optional[Union[str, Path]],
    method: str = "wls"
) -> FitResult:
    df = load_ct(data_csv)
    df = _mask_profile(df, profile_yaml)
    x, y, w_meas = _prepare_xyw(df)
    X = np.vstack([x, np.ones_like(x)]).T

    if method == "wls":
        beta = _wls(X, y, w_meas)
        yhat = X @ beta
        return FitResult(float(beta[0]), float(beta[1]), x, y, w_meas, yhat)

    if method == "huber":
        beta, w_eff = _irls_huber(X, y, w_meas, c=1.345)
        yhat = X @ beta
        return FitResult(float(beta[0]), float(beta[1]), x, y, w_eff, yhat)

    # fallback: OLS
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    return FitResult(float(beta[0]), float(beta[1]), x, y, w_meas, yhat)

# ===== 固定斜率：只估截距（log10 A），並提供一側／兩側上限 =====

@dataclass
class FixedSlopeResult:
    slope_fixed: float
    intercept_log10: float
    intercept_se: float
    A_hat: float
    A_ul: float  # one-sided (1-alpha) UL in linear A
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray

def _wls_fixed_intercept(lx: np.ndarray, ly: np.ndarray, w: np.ndarray, m_fixed: float) -> Tuple[float, float]:
    """
    在 ly = m_fixed * lx + b 中，以 WLS 解 b。
    b_hat = sum(w * (ly - m lx)) / sum(w)
    var(b_hat) ≈ 1 / sum(w)  （測量誤差已由 w 表示）
    回傳 (b_hat, se_b)
    """
    wx = np.clip(w, _EPS, None)
    num = np.sum(wx * (ly - m_fixed * lx))
    den = np.sum(wx)
    b_hat = num / max(den, _EPS)

    # two choices: (i) known-variance (1/sum w) or (ii) sandwich 調整
    # 這裡提供較穩健的 sandwich 版本（依殘差大小調整）
    r = (ly - (m_fixed * lx + b_hat))
    s2 = np.sum(wx * r * r) / max((lx.size - 1), 1)  # 殘差調整
    se_b_known = np.sqrt(1.0 / max(den, _EPS))
    se_b = max(se_b_known, np.sqrt(s2 / max(den, _EPS)))
    return float(b_hat), float(se_b)

def _norm_ppf(p: float) -> float:
    """ 近似標準常態分佈 inverse CDF（Acklam approximation） """
    # 參考: https://web.archive.org/web/20150910044729/http://home.online.no/~pjacklam/notes/invnorm/
    a = [ -3.969683028665376e+01,  2.209460984245205e+02,
          -2.759285104469687e+02,  1.383577518672690e+02,
          -3.066479806614716e+01,  2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,  1.615858368580409e+02,
          -1.556989798598866e+02,  6.680131188771972e+01,
          -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03, -3.223964580411365e-01,
          -2.400758277161838e+00, -2.549732539343734e+00,
           4.374664141464968e+00,  2.938163982698783e+00 ]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00 ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = np.sqrt(-2*np.log(p))
        num = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
        den = ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        return num/den
    if phigh < p:
        q = np.sqrt(-2*np.log(1-p))
        num = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
        den = ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        return num/den
    q = p-0.5
    r = q*q
    num = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q
    den = (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    return num/den

def do_fit_fixed_slope(
    data_csv: Path,
    profile_yaml: Optional[Union[str, Path]],
    m_fixed: float = 2.0,
    alpha: float = 0.05,
) -> FixedSlopeResult:
    df = load_ct(data_csv)
    df = _mask_profile(df, profile_yaml)
    x, y, w = _prepare_xyw(df)
    lx = x; ly = y

    b_hat, se_b = _wls_fixed_intercept(lx, ly, w, m_fixed)
    z = _norm_ppf(1.0 - alpha)  # one-sided
    b_ul = b_hat + z * se_b

    A_hat = 10.0**b_hat
    A_ul = 10.0**b_ul

    return FixedSlopeResult(
        slope_fixed=float(m_fixed),
        intercept_log10=float(b_hat),
        intercept_se=float(se_b),
        A_hat=float(A_hat),
        A_ul=float(A_ul),
        x=lx, y=ly, w=w
    )
