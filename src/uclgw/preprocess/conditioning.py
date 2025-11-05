# src/uclgw/preprocess/conditioning.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

def _welch_psd(x: np.ndarray, fs: float, nperseg: int, noverlap: int | None):
    n = x.size
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap
    if step <= 0 or nperseg > n:
        nperseg = min(n, nperseg)
        noverlap = nperseg // 2
        step = max(1, nperseg - noverlap)

    win = np.hanning(nperseg)
    U = (win**2).sum()
    idx = 0
    acc = None
    m = 0
    while idx + nperseg <= n:
        seg = x[idx:idx+nperseg] * win
        X = np.fft.rfft(seg)
        Pxx = (1.0/(fs*U)) * (np.abs(X)**2) * 2.0  # one-sided
        if acc is None:
            acc = Pxx
        else:
            acc += Pxx
        m += 1
        idx += step
    if m == 0:
        # fallback: single segment
        seg = x[:nperseg] * np.hanning(nperseg)
        X = np.fft.rfft(seg)
        Pxx = (1.0/(fs*U)) * (np.abs(X)**2) * 2.0
        acc = Pxx
        m = 1
    Pxx_avg = acc / m
    f = np.fft.rfftfreq(nperseg, d=1.0/fs)
    return f, Pxx_avg

def _gate_outliers(z, zthr=5.0):
    if zthr is None or zthr <= 0:
        return z.copy()
    x = z.copy()
    mu = np.median(x)
    sig = np.std(x)
    bad = np.where(np.abs((x - mu) / (sig + 1e-12)) > zthr)[0]
    if bad.size == 0:
        return x
    # simple replacement with local median
    for i in bad:
        lo = max(0, i-10); hi = min(x.size, i+10)
        x[i] = np.median(x[lo:hi])
    return x

def _whiten_full_series(h: np.ndarray, fs: float, f_psd: np.ndarray, pxx: np.ndarray):
    # interpolate PSD onto rFFT bins of full series, divide in freq-domain
    n = h.size
    H = np.fft.rfft(h)
    f_bins = np.fft.rfftfreq(n, d=1.0/fs)
    pxx_interp = np.interp(f_bins, f_psd, pxx, left=pxx[0], right=pxx[-1])
    amp = np.sqrt(np.maximum(pxx_interp, 1e-40))
    W = H / amp
    w = np.fft.irfft(W, n=n)
    # normalize to unit std
    w = (w - np.mean(w)) / (np.std(w) + 1e-12)
    return w

def prepare_event(seg_json: dict, psd_dir: Path, white_dir: Path,
                  nperseg: int = 4096*4, noverlap: int | None = None,
                  gate_z: float = 5.0):
    psd_dir.mkdir(parents=True, exist_ok=True)
    white_dir.mkdir(parents=True, exist_ok=True)
    out = {"psd": [], "whitened": []}

    for s in seg_json["segments"]:
        raw_path = Path(s["path_raw"])
        dat = np.load(raw_path, allow_pickle=True)
        t = dat["t"]
        h = dat["h"]
        meta = json.loads(str(dat["meta"]))

        # gating
        h_g = _gate_outliers(h, zthr=gate_z)
        # PSD
        f_psd, pxx = _welch_psd(h_g, fs=meta["fs"], nperseg=nperseg, noverlap=noverlap)
        # save PSD
        psd_path = psd_dir / f"{meta['event']}_{meta['ifo']}.csv"
        pd.DataFrame({"freq_hz": f_psd, "psd": pxx}).to_csv(psd_path, index=False)
        out["psd"].append(str(psd_path))

        # whiten
        w = _whiten_full_series(h_g, fs=meta["fs"], f_psd=f_psd, pxx=pxx)
        white_path = white_dir / f"{meta['event']}_{meta['ifo']}.npz"
        np.savez_compressed(white_path, t=t, w=w, meta=json.dumps(meta))
        out["whitened"].append(str(white_path))

    return out
