# src/uclgw/features/network_timing.py
from __future__ import annotations
from pathlib import Path
import itertools, json
import numpy as np

C = 299792458.0  # m/s

# 近似地表大圓距離（公尺）；只作 sanity bound 用，不進主擬合
BASELINES_M = {
    ("H1", "L1"): 3002e3,  # Hanford-Livingston ~ 3002 km
    ("H1", "V1"): 8670e3,  # Hanford-Virgo    ~ 8671 km
    ("L1", "V1"): 7610e3,  # Livingston-Virgo ~ 7612 km
}

def _xcorr_argmax(a: np.ndarray, b: np.ndarray) -> int:
    """Return lag index (b relative to a) that maximizes correlation (FFT-based)."""
    n = max(len(a), len(b))
    nfft = int(2**np.ceil(np.log2(2*n)))
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    x = np.fft.irfft(A * np.conj(B), nfft)
    # circular shift; take the max over full range and convert to signed lag
    k = np.argmax(x)
    if k > nfft//2:
        k -= nfft
    return int(k)

def estimate_delays_and_bounds(white_paths: list[Path]) -> dict:
    """Estimate pairwise delays (sec) and crude c_T bounds."""
    # load
    series = {}
    metas = {}
    for p in white_paths:
        d = np.load(p, allow_pickle=True)
        w = d["w"].astype(float)
        meta = json.loads(str(d["meta"]))
        ifo = meta["ifo"]
        fs = float(meta["fs"])
        series[ifo] = w
        metas[ifo] = meta

    results = {"pairs": []}
    ifos = sorted(series.keys())
    for a, b in itertools.combinations(ifos, 2):
        wa, wb = series[a], series[b]
        fs = float(metas[a]["fs"])
        lag_samp = _xcorr_argmax(wa, wb)
        dt = lag_samp / fs  # seconds; wb arrives dt after wa (approx)
        key = tuple(sorted((a, b)))
        d = BASELINES_M.get(key, None)
        v_est = None
        eps = None
        if d is not None and abs(dt) > 0:
            v_est = abs(d / dt)              # m/s
            eps = (v_est / C) - 1.0          # fractional
        results["pairs"].append({
            "ifo_a": a, "ifo_b": b,
            "lag_samples": lag_samp, "dt_sec": dt,
            "baseline_m": d, "v_over_c_minus_1": eps
        })
    return results
