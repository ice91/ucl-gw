# src/uclgw/features/dispersion_fit.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

C_LIGHT = 299792458.0

# ---------- helper: load whitened ----------
def _load_white(work_dir: Path, event: str):
    # expects files data/work/whitened/{event}_{IFO}.npz
    out = {}
    for ifo in ("H1", "L1", "V1"):
        p = work_dir / f"{event}_{ifo}.npz"
        if not p.exists():
            continue
        dat = np.load(p, allow_pickle=True)
        t = dat["t"]; w = dat["w"]
        meta = json.loads(str(dat["meta"]))
        out[ifo] = dict(t=t, w=w, fs=meta["fs"], meta=meta)
    if not out:
        raise RuntimeError(f"No whitened files found for {event} in {work_dir}")
    return out

# ---------- helper: frequency grid bins ----------
def _freq_bins(fmin: float, fmax: float, n_bins: int):
    edges = np.geomspace(fmin, fmax, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return edges, centers

# ---------- proxy-k2 (既有) ----------
def proxy_k2_points(event: str, fmin: float, fmax: float, n_bins: int, work_dir: Path) -> pd.DataFrame:
    _, fcs = _freq_bins(fmin, fmax, n_bins)
    k = 2.0 * np.pi * fcs / C_LIGHT
    # 只為了 slope（截距不重要），直接給 δc_T^2 ∝ k^2
    y = (k**2)
    # 估計 sigma：常數權重（WLS 等效 OLS）
    sigma = np.full_like(y, 1e-2)
    rows = []
    for ifo in ("H1", "L1", "V1"):
        for fi, ki, yi, si in zip(fcs, k, y, sigma):
            rows.append(dict(event=event, ifo=ifo, f_hz=float(fi),
                             k=float(ki), delta_ct2=float(yi), sigma=float(si)))
    return pd.DataFrame(rows)

# ---------- NEW: phase-fit via cross-spectrum ----------
def _welch_csd(x: np.ndarray, y: np.ndarray, fs: float, nperseg: int = 8192, noverlap: int | None = None):
    if noverlap is None:
        noverlap = nperseg // 2
    n = min(x.size, y.size)
    step = nperseg - noverlap
    win = np.hanning(nperseg)
    U = (win**2).sum()
    i = 0; M = 0
    Sxy = None; Sxx = None; Syy = None
    while i + nperseg <= n:
        xs = x[i:i+nperseg] * win
        ys = y[i:i+nperseg] * win
        X = np.fft.rfft(xs); Y = np.fft.rfft(ys)
        _Sxy = (X * np.conj(Y)) * (2.0/(fs*U))
        _Sxx = (np.abs(X)**2) * (2.0/(fs*U))
        _Syy = (np.abs(Y)**2) * (2.0/(fs*U))
        if Sxy is None:
            Sxy, Sxx, Syy = _Sxy, _Sxx, _Syy
        else:
            Sxy += _Sxy; Sxx += _Sxx; Syy += _Syy
        M += 1; i += step
    if M == 0:
        raise RuntimeError("window too large for series")
    Sxy /= M; Sxx /= M; Syy /= M
    f = np.fft.rfftfreq(nperseg, d=1.0/fs)
    # 相干度
    gamma2 = (np.abs(Sxy)**2) / (np.maximum(Sxx, 1e-30) * np.maximum(Syy, 1e-30))
    gamma2 = np.clip(gamma2, 0.0, 1.0)
    # 相位
    phi = np.unwrap(np.angle(Sxy))
    return f, phi, gamma2, M

def _fit_linear(x, y, w=None):
    X = np.vstack([x, np.ones_like(x)]).T
    if w is None:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta[0], beta[1]
    W = np.diag(w)
    XtW = X.T @ W
    beta = np.linalg.inv(XtW @ X) @ (XtW @ y)
    return beta[0], beta[1]

def _phasefit_pair(xA, xB, fs, fmin, fmax, n_bins, null_mode="none"):
    # 可選 null：循環位移其中一路，破壞跨站相干
    if null_mode == "timeshift":
        shift = np.random.randint(int(0.2*fs), int(0.4*fs))  # 0.2~0.4s
        xB = np.roll(xB, shift)

    f, phi, gamma2, M = _welch_csd(xA, xB, fs=fs, nperseg=8192, noverlap=4096)
    # 只取頻窗
    mwin = (f >= fmin) & (f <= fmax)
    f = f[mwin]; phi = phi[mwin]; gamma2 = gamma2[mwin]
    # 拿掉「整體到達時差」：phi ≈ 2π f τ + ϕ_res。線性擬合後取殘差 |ϕ_res|
    a, b = _fit_linear(f, phi, w=gamma2)  # phi ~ a f + b
    phi_res = phi - (a*f + b)
    # 以頻帶等比分箱
    edges, centers = _freq_bins(fmin, fmax, n_bins)
    rows = []
    for i in range(len(edges)-1):
        mask = (f>=edges[i]) & (f<edges[i+1])
        if not np.any(mask): 
            continue
        # bin 內的 proxy：|phi_res| 代表色散相位偏差強度；σ_phi 由相干度近似
        # var(phi) ≈ (1-γ²)/(2Mγ²)
        g = np.clip(np.median(gamma2[mask]), 1e-6, 1-1e-6)
        var_phi = (1.0 - g) / (2.0 * M * g)
        sigma_phi = np.sqrt(max(var_phi, 1e-12))
        y = float(np.median(np.abs(phi_res[mask]))) + 1e-16
        ff = float(centers[i])
        k = 2.0*np.pi*ff/C_LIGHT
        rows.append((ff, k, y, sigma_phi))
    return rows

def phasefit_points(event: str, fmin: float, fmax: float, n_bins: int,
                    work_dir: Path, null_mode: str = "none") -> pd.DataFrame:
    W = _load_white(work_dir, event)
    have = sorted(W.keys())
    pairs = []
    if "H1" in have and "L1" in have: pairs.append(("H1","L1"))
    if "H1" in have and "V1" in have: pairs.append(("H1","V1"))
    if "L1" in have and "V1" in have: pairs.append(("L1","V1"))
    rows = []
    for A,B in pairs:
        pa = _phasefit_pair(W[A]["w"], W[B]["w"], fs=W[A]["fs"],
                            fmin=fmin, fmax=fmax, n_bins=n_bins, null_mode=null_mode)
        for (ff, k, y, s) in pa:
            rows.append(dict(event=event, ifo=f"{A}-{B}", f_hz=ff, k=k,
                             delta_ct2=y, sigma=s))
    if not rows:
        raise RuntimeError("No IFO pairs available for phase-fit.")
    return pd.DataFrame(rows)
