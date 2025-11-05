# src/uclgw/features/network_timing.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

C_LIGHT = 299792458.0  # m/s

# 粗略基線（公里；用於 light-time 合理性檢查，非精密推論）
_BASELINE_KM = {
    ("H1", "L1"): 3002.0,   # Hanford-Livingston
    ("H1", "V1"): 8670.0,   # Hanford-Virgo
    ("L1", "V1"): 7618.0,   # Livingston-Virgo
}

def _load_white_npz(work_dir: Path, event: str, ifo: str):
    p = work_dir / f"{event}_{ifo}.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    dat = np.load(p, allow_pickle=True)
    t = dat["t"]
    w = dat["w"]
    meta = json.loads(str(dat["meta"]))
    fs = float(meta["fs"])
    return t, w, fs

def _xcorr_fft(a: np.ndarray, b: np.ndarray):
    # 以 FFT 做互相關，回傳相關值與對應的樣本位移（lag）
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    r = np.fft.irfft(A * np.conj(B), nfft)
    # 對齊：lags = -(len(b)-1) ... (len(a)-1)
    r = np.roll(r, len(b) - 1)
    lags = np.arange(-(len(b) - 1), len(a))
    return r[:lags.size], lags

def _lag_peak(a: np.ndarray, b: np.ndarray, fs: float):
    r, lags = _xcorr_fft(a, b)
    k = int(np.argmax(np.abs(r)))
    lag_samp = int(lags[k])
    tau = lag_samp / fs
    # 粗略 ±1 sample 當作區間（只做 sanity）
    ci = ((lag_samp - 1) / fs, (lag_samp + 1) / fs)
    return float(tau), float(ci[0]), float(ci[1]), float(r[k])

def network_delay_bounds(event: str, work_dir: Path) -> dict:
    """
    給每一對 IFO 做到達時差的粗估（以 whitened 資料的互相關峰值），
    並與基線的光行時間做「合理性」檢查。這是 side-car JSON，不參與擬合。
    """
    have = [ifo for ifo in ("H1", "L1", "V1") if (work_dir / f"{event}_{ifo}.npz").exists()]
    if not have:
        return {"event": event, "pairs": {}, "fs": None}

    series = {}
    fs = None
    for ifo in have:
        _, w, fs0 = _load_white_npz(work_dir, event, ifo)
        series[ifo] = w
        fs = fs or fs0

    out = {"event": event, "fs": fs, "pairs": {}}

    def _eval_pair(a: str, b: str):
        if a not in series or b not in series:
            return
        tau, lo, hi, rpk = _lag_peak(series[a], series[b], fs)
        base_km = _BASELINE_KM.get(tuple(sorted((a, b))), None)
        if base_km is not None:
            lt = (base_km * 1000.0) / C_LIGHT  # 光行時間（秒）
            # 給寬鬆 3ms 邊際：來源方向未知；只看「大致合理」
            consistent = (abs(tau) <= lt + 0.003)
        else:
            lt = None
            consistent = True
        out["pairs"][f"{a}-{b}"] = {
            "lag_s": tau,
            "lag_ci_s": [lo, hi],
            "peak_corr": float(rpk),
            "baseline_lighttime_s": lt,
            "consistent": bool(consistent)
        }

    if "H1" in series and "L1" in series: _eval_pair("H1", "L1")
    if "H1" in series and "V1" in series: _eval_pair("H1", "V1")
    if "L1" in series and "V1" in series: _eval_pair("L1", "V1")

    return out
