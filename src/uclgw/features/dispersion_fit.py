# src/uclgw/features/dispersion_fit.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import json

C = 299792458.0  # m/s
TINY = 1e-30

def _periodogram(w: np.ndarray, fs: float):
    n = w.size
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, d=1.0/fs)
    # white series → 平坦，但仍可作權重
    P = (np.abs(W)**2) / max(TINY, n)
    return f, P

def _choose_bins_logspace(f, P, fmin, fmax, n_bins):
    mask = (f >= fmin) & (f <= fmax)
    f_sel = f[mask]
    P_sel = P[mask]
    if f_sel.size < n_bins:
        n_bins = max(4, f_sel.size)
    edges = np.logspace(np.log10(max(fmin, 1e-3)), np.log10(fmax), n_bins+1)
    centers = np.sqrt(edges[:-1]*edges[1:])
    idxs = []
    amp = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f_sel >= lo) & (f_sel < hi)
        if np.any(m):
            # 取能量最大的頻點代表該窗；也可改中位數
            j = np.argmax(P_sel[m])
            pick = np.flatnonzero(m)[j]
            idxs.append(pick)
            amp.append(P_sel[pick])
    return f_sel[idxs], np.array(amp)

def _proxy_k2_points(f_points: np.ndarray, amps: np.ndarray, alpha: float = 1e-20):
    # 以 k0 為幾何平均頻率；delta_cT^2 = alpha * (k/k0)^2
    k = 2*np.pi*f_points / C
    k0 = np.exp(np.mean(np.log(np.maximum(k, 1e-30))))
    delta = alpha * (np.maximum(k, 1e-30)/k0)**2
    # 簡單不確定度模型：幅度權重的 1/sqrt 對應
    sigma = 0.3 * delta * (np.median(amps) / np.maximum(amps, 1e-30))**0.5
    return k, delta, sigma

def build_ct_points_from_whitened(white_paths: list[Path], fmin: float, fmax: float,
                                  n_bins: int = 24, mode: str = "proxy-k2") -> pd.DataFrame:
    rows = []
    for p in white_paths:
        d = np.load(p, allow_pickle=True)
        w = d["w"].astype(float)
        meta = json.loads(str(d["meta"]))
        fs = float(meta["fs"])
        event = meta["event"]; ifo = meta["ifo"]

        f, P = _periodogram(w, fs)
        f_pts, amps = _choose_bins_logspace(f, P, fmin, fmax, n_bins)

        if mode == "proxy-k2":
            k, delta, sig = _proxy_k2_points(f_pts, amps, alpha=1e-20)
        else:
            raise NotImplementedError(f"mode={mode} not implemented")

        for fi, ki, di, si in zip(f_pts, k, delta, sig):
            rows.append({
                "event": event, "ifo": ifo,
                "f_hz": float(fi), "k": float(ki),
                "delta_ct2": float(di), "sigma": float(si),
            })
    return pd.DataFrame(rows)
