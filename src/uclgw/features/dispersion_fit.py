# src/uclgw/features/dispersion_fit.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

C_LIGHT = 299_792_458.0

# ------------------------
# 工具：Welch 版 CSD/Coherence
# ------------------------
def _welch_csd_xy(x: np.ndarray, y: np.ndarray, fs: float, nperseg: int = 4096, noverlap: int | None = None):
    if noverlap is None:
        noverlap = nperseg // 2
    step = max(1, nperseg - noverlap)
    n = min(x.size, y.size)
    nperseg = min(nperseg, n)

    win = np.hanning(nperseg)
    U = (win**2).sum()

    acc_cxy = acc_sxx = acc_syy = None
    m = 0
    for idx in range(0, n - nperseg + 1, step):
        xs = x[idx:idx+nperseg] * win
        ys = y[idx:idx+nperseg] * win
        X = np.fft.rfft(xs); Y = np.fft.rfft(ys)
        Sxx = (2.0/(fs*U)) * (np.abs(X)**2)
        Syy = (2.0/(fs*U)) * (np.abs(Y)**2)
        Cxy = (2.0/(fs*U)) * (X * np.conj(Y))
        if acc_cxy is None:
            acc_cxy, acc_sxx, acc_syy = Cxy, Sxx, Syy
        else:
            acc_cxy += Cxy; acc_sxx += Sxx; acc_syy += Syy
        m += 1

    if m == 0:
        xs = x[:nperseg] * win
        ys = y[:nperseg] * win
        X = np.fft.rfft(xs); Y = np.fft.rfft(ys)
        Sxx = (2.0/(fs*U)) * (np.abs(X)**2)
        Syy = (2.0/(fs*U)) * (np.abs(Y)**2)
        Cxy = (2.0/(fs*U)) * (X * np.conj(Y))
        acc_cxy, acc_sxx, acc_syy = Cxy, Sxx, Syy
        m = 1

    Cxy_avg = acc_cxy / m
    Sxx_avg = acc_sxx / m
    Syy_avg = acc_syy / m
    f = np.fft.rfftfreq(nperseg, d=1.0/fs)
    return f, Cxy_avg, Sxx_avg, Syy_avg


def _timeshift(a: np.ndarray, shift_samples: int) -> np.ndarray:
    if shift_samples == 0 or a.size == 0: return a.copy()
    s = shift_samples % a.size
    if s == 0: return a.copy()
    return np.concatenate([a[-s:], a[:-s]])


def _weighted_linfit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x); y = np.asarray(y); w = np.asarray(w)
    if x.size == 0:
        return 0.0, 0.0
    W = np.sqrt(np.maximum(w, 1e-16))
    A = np.stack([np.ones_like(x), x], axis=1)
    Aw = A * W[:, None]; yw = y * W
    beta, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    a0, a1 = beta[0], beta[1]
    return float(a0), float(a1)


def _logspace_edges(fmin: float, fmax: float, n_bins: int) -> np.ndarray:
    return np.logspace(np.log10(fmin), np.log10(fmax), n_bins + 1)


def _detect_ifos(white_dir: Path, event: str) -> List[str]:
    ifos = []
    for p in sorted((white_dir).glob(f"{event}_*.npz")):
        tag = p.stem.split("_")[-1]
        ifos.append(tag)
    return sorted(list(set(ifos)))


def _load_whitened(work_dir: Path, event: str) -> Dict[str, Dict]:
    d: Dict[str, Dict] = {}
    for p in sorted((work_dir).glob(f"{event}_*.npz")):
        with np.load(p, allow_pickle=True) as z:
            t = z["t"]; w = z["w"]; meta = json.loads(str(z["meta"]))
        ifo = p.stem.split("_")[-1]
        d[ifo] = {"t": t, "w": w, "meta": meta}
    if not d:
        raise FileNotFoundError(f"no whitened npz found under {work_dir} for event={event}")
    return d

# ------------------------
# 代理：proxy-k2（輸出標準欄位；sigma=1.0 等權）
# ------------------------
def proxy_k2_points(event: str, fmin: float, fmax: float, n_bins: int, work_dir: Path) -> pd.DataFrame:
    ifos = _detect_ifos(work_dir, event)
    edges = _logspace_edges(fmin, fmax, n_bins)
    rows = []
    for ifo in ifos:
        for i in range(n_bins):
            flo, fhi = edges[i], edges[i+1]
            fmid = np.sqrt(flo * fhi)
            k = 2.0 * np.pi * fmid / C_LIGHT
            delta_ct2 = (k**2)  # 理想 NLO 斜率=2
            rows.append({
                "event": event, "ifo": ifo, "f_hz": fmid, "k": k,
                "delta_ct2": float(delta_ct2), "sigma": 1.0  # 等權
            })
    return pd.DataFrame(rows)

# ------------------------
# 真實：phase-fit（穩健 detrend + 加權 bin 斜率；sigma=1.0 等權）
# ------------------------
def phasefit_points(
    event: str,
    fmin: float,
    fmax: float,
    n_bins: int,
    work_dir: Path,
    null_mode: str = "none",
    nperseg: int = 8192,
    noverlap: int | None = None,
    coherence_min: float = 0.85,
    min_bins_count: int = 6,
    drop_edge_bins: int = 2,
) -> pd.DataFrame:

    W = _load_whitened(work_dir, event)
    ifos = sorted(W.keys())
    if len(ifos) < 2:
        raise RuntimeError(f"need >=2 IFOs, got {ifos}")

    fs_ref = None
    for ifo in ifos:
        fs = float(W[ifo]["meta"]["fs"])
        if fs_ref is None: fs_ref = fs
        if abs(fs - fs_ref) > 1e-9:
            raise RuntimeError("sampling rate mismatch across IFOs")

    pairs = [(ifos[i], ifos[j]) for i in range(len(ifos)) for j in range(i+1, len(ifos))]
    edges = _logspace_edges(fmin, fmax, n_bins)

    pair_bins: Dict[tuple[str, str], Dict[str, np.ndarray]] = {}

    for (a, b) in pairs:
        xa = W[a]["w"].copy()
        yb = W[b]["w"].copy()
        if null_mode == "timeshift":
            yb = _timeshift(yb, shift_samples=(len(yb) // 4))

        f, Cxy, Sxx, Syy = _welch_csd_xy(xa, yb, fs=fs_ref, nperseg=nperseg, noverlap=noverlap)
        coh2 = np.clip((np.abs(Cxy)**2) / (np.maximum(Sxx,1e-30)*np.maximum(Syy,1e-30)), 0.0, 1.0)
        phi = np.unwrap(np.angle(Cxy))

        # ---- 穩健 detrend：只用中段 + 高相干 × 高頻加權；做一次 outlier 修正 ----
        mband = (f >= fmin) & (f <= fmax)
        mcoh = (coh2 >= coherence_min)
        mid  = (f >= fmin*1.2) & (f <= fmax*0.9)

        fref = max(60.0, fmin)
        w_detr = coh2 * (np.clip(f / fref, 0.5, 10.0) ** 2)

        use = mband & mid & mcoh
        if not np.any(use):
            use = mband & mid
        if not np.any(use):
            use = mband

        a0, a1 = _weighted_linfit(f[use], phi[use], w_detr[use])
        phi_res = phi - (a0 + a1 * f)

        # one-shot outlier trim（90 百分位）
        use2 = use.copy()
        try:
            thr = np.percentile(np.abs(phi_res[use]), 90.0)
            use2 = use & (np.abs(phi_res) <= thr)
        except Exception:
            pass
        if np.any(use2):
            a0, a1 = _weighted_linfit(f[use2], phi[use2], w_detr[use2])
            phi_res = phi - (a0 + a1 * f)

        # ---- 逐 bin 估 |dφ/df|，保留輸出長度；邊界 bin 以 NaN/0 佔位 ----
        k_arr  = np.full(n_bins, np.nan, dtype=float)
        y_arr  = np.full(n_bins, np.nan, dtype=float)
        w_arr  = np.zeros(n_bins, dtype=float)
        fmid_arr = np.full(n_bins, np.nan, dtype=float)

        lo_i = int(max(0, drop_edge_bins))
        hi_i = int(max(lo_i, n_bins - drop_edge_bins))

        for i in range(n_bins):
            flo, fhi = edges[i], edges[i+1]
            fmid = np.sqrt(flo * fhi)

            if not (lo_i <= i < hi_i):
                # 邊界 bin：保留 NaN/0.0 以維持等長
                fmid_arr[i] = fmid
                continue

            sel = (f >= flo) & (f < fhi) & (coh2 >= coherence_min)
            if np.sum(sel) < 5:
                sel = (f >= flo) & (f < fhi)
                if np.sum(sel) < 5:
                    fmid_arr[i] = fmid
                    continue

            wloc = coh2 * (np.clip(f / fref, 0.5, 10.0) ** 2)
            a_loc, b_loc = _weighted_linfit(f[sel], phi_res[sel], wloc[sel])  # b_loc ≈ dφ/df
            ybin = abs(b_loc)
            kbin = 2.0 * np.pi * fmid / C_LIGHT
            wbin = float(np.sum(wloc[sel]))

            k_arr[i] = kbin
            y_arr[i] = ybin
            w_arr[i] = wbin
            fmid_arr[i] = fmid

        pair_bins[(a, b)] = {"k": k_arr, "y": y_arr, "w": w_arr, "fmid": fmid_arr}

    # ---- IFO 聚合（權重平均），sigma 等權 1.0；不足 bin 會自動被跳過 ----
    rows = []
    for ifo in ifos:
        for ib in range(n_bins):
            ys, ws, ks, fs = [], [], [], []
            for (a, b), dump in pair_bins.items():
                if ifo not in (a, b): continue
                yb = dump["y"][ib]; wb = dump["w"][ib]; kb = dump["k"][ib]; fm = dump["fmid"][ib]
                if not np.isfinite(yb) or wb <= 0.0 or not np.isfinite(kb) or not np.isfinite(fm):
                    continue
                ys.append(yb); ws.append(wb); ks.append(kb); fs.append(fm)
            if len(ys) == 0:
                continue

            ys = np.asarray(ys); ws = np.asarray(ws); ks = np.asarray(ks); fs = np.asarray(fs)
            y_agg = float(np.sum(ws * ys) / np.sum(ws))
            k_agg = float(np.sum(ws * ks) / np.sum(ws))
            f_agg = float(np.sum(ws * fs) / np.sum(ws))

            rows.append({
                "event": event, "ifo": ifo, "f_hz": f_agg, "k": k_agg,
                "delta_ct2": max(y_agg, 1e-18),
                "sigma": 1.0  # 等權；WLS 端用 1/sigma^2
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ok = df.groupby("ifo")["k"].count() >= min_bins_count
    keep = set(ok[ok].index.tolist())
    df = df[df["ifo"].isin(keep)].reset_index(drop=True)
    return df
