# src/uclgw/features/dispersion_fit.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

C_LIGHT = 299_792_458.0

# ------------------------
# 基礎工具：Welch 版 CSD/Coherence
# ------------------------
def _welch_csd_xy(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    nperseg: int = 4096,
    noverlap: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    回傳:
      f: 頻率 (Hz), Cxy: 平均互譜 (complex), Sxx, Syy: 各自平均功率譜 (real)
    """
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap
    n = min(x.size, y.size)
    nperseg = min(nperseg, n)
    if step <= 0:
        step = max(1, nperseg // 2)

    win = np.hanning(nperseg)
    U = (win**2).sum()

    acc_cxy = None
    acc_sxx = None
    acc_syy = None
    m = 0
    idx = 0
    while idx + nperseg <= n:
        xs = x[idx:idx+nperseg] * win
        ys = y[idx:idx+nperseg] * win
        X = np.fft.rfft(xs)
        Y = np.fft.rfft(ys)
        Sxx = (2.0/(fs*U)) * (np.abs(X)**2)
        Syy = (2.0/(fs*U)) * (np.abs(Y)**2)
        Cxy = (2.0/(fs*U)) * (X * np.conj(Y))
        if acc_cxy is None:
            acc_cxy = Cxy
            acc_sxx = Sxx
            acc_syy = Syy
        else:
            acc_cxy += Cxy
            acc_sxx += Sxx
            acc_syy += Syy
        m += 1
        idx += step

    if m == 0:
        # 單窗 fallback
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
    """循環位移，用來破壞跨台站相干（實驗 null）"""
    if shift_samples == 0:
        return a.copy()
    shift = shift_samples % a.size
    if shift == 0:
        return a.copy()
    return np.concatenate([a[-shift:], a[:-shift]])


def _weighted_linfit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """
    加權最小平方，擬合 y ≈ a0 + a1 * x
    回傳 a0, a1
    """
    w = np.asarray(w)
    x = np.asarray(x); y = np.asarray(y)
    W = np.sqrt(np.maximum(w, 1e-16))
    A = np.stack([np.ones_like(x), x], axis=1)
    Aw = A * W[:, None]
    yw = y * W
    beta, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    a0, a1 = beta[0], beta[1]
    return float(a0), float(a1)


def _logspace_edges(fmin: float, fmax: float, n_bins: int) -> np.ndarray:
    return np.logspace(np.log10(fmin), np.log10(fmax), n_bins + 1)


def _bin_rms(values: np.ndarray, weights: np.ndarray) -> float:
    """穩健：用加權 RMS (避免單點極端值)，作為頻帶代表 y"""
    w = np.maximum(np.asarray(weights), 0.0)
    if values.size == 0 or np.sum(w) <= 0:
        return np.nan
    return float(np.sqrt(np.sum(w * (values**2)) / np.sum(w)))


# ------------------------
# 既有的 proxy-k2：保留
# ------------------------
def proxy_k2_points(
    event: str,
    fmin: float,
    fmax: float,
    n_bins: int,
    work_dir: Path,
) -> pd.DataFrame:
    """
    玩具/代理：直接用 y ∝ k^2 做煙霧測試（保持既有行為以利對照）。
    """
    ifo_list = _detect_ifos(work_dir, event)
    edges = _logspace_edges(fmin, fmax, n_bins)
    rows = []
    for ifo in ifo_list:
        for i in range(n_bins):
            flo, fhi = edges[i], edges[i+1]
            fmid = np.sqrt(flo * fhi)
            k = 2.0 * np.pi * fmid / C_LIGHT
            y = (k**2)  # 代理：讓 slope=2 成立
            w = 1.0
            rows.append({"event": event, "ifo": ifo, "k": k, "y": y, "w": w})
    return pd.DataFrame(rows)


# ------------------------
# 新：phase-fit（真實資料）
# ------------------------
def phasefit_points(
    event: str,
    fmin: float,
    fmax: float,
    n_bins: int,
    work_dir: Path,
    null_mode: str = "none",
    nperseg: int = 4096,
    noverlap: int | None = None,
    coherence_min: float = 0.6,
    min_bins_count: int = 6,
) -> pd.DataFrame:
    """
    真實資料的相位擬合：
      1) 對每對 IFO 計互譜 Cxy 與 coherence γ^2(f)
      2) 取 arg(Cxy) 解包相位，做加權（γ^2）線性去趨勢（扣 a0+a1*f）
      3) 將殘差相位 φ_res(f) 在 [fmin,fmax] 用 log-space 分成 n_bins
      4) 對各 bin 以加權 RMS(|φ_res|) 作代表量 y_bin；k_bin = 2π f_mid / c
      5) 將 pair 結果聚合回各 IFO：對含該 IFO 的所有 pair 在同一 bin 做加權平均
      6) 產出欄位(event, ifo, k, y, w)，可直接供 Slope-2 擬合（只看斜率）

    註：這裡的 y 是「相位殘差的無單位 proxy」（rad 的加權 RMS），斜率驗證
        聚焦於預言 s≈2；截距留待進一步物理常數的映射。
    """
    # 讀取 whitened 資料
    W = _load_whitened(work_dir, event)
    ifos = sorted(W.keys())
    if len(ifos) < 2:
        raise RuntimeError(f"need >=2 IFOs, got {ifos}")

    # 產 pair 名單
    pairs = []
    for i in range(len(ifos)):
        for j in range(i+1, len(ifos)):
            pairs.append((ifos[i], ifos[j]))

    # 計算每個 pair 的 (k_bin, y_bin, w_bin)
    pair_bins: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for (a, b) in pairs:
        ta, wa, meta_a = W[a]["t"], W[a]["w"], W[a]["meta"]
        tb, wb, meta_b = W[b]["t"], W[b]["w"], W[b]["meta"]
        if abs(meta_a["fs"] - meta_b["fs"]) > 1e-9:
            raise RuntimeError("sampling rate mismatch between IFOs")

        fs = float(meta_a["fs"])
        xa = wa.copy()
        yb = wb.copy()
        # optional null：打亂其中一路以破壞相干
        if null_mode == "timeshift":
            yb = _timeshift(yb, shift_samples=(len(yb) // 4))

        f, Cxy, Sxx, Syy = _welch_csd_xy(xa, yb, fs=fs, nperseg=nperseg, noverlap=noverlap)
        # coherence
        coh2 = (np.abs(Cxy)**2) / (np.maximum(Sxx, 1e-30) * np.maximum(Syy, 1e-30))
        coh2 = np.clip(coh2, 0.0, 1.0)

        # 頻率視窗 + 相干門檻
        mband = (f >= fmin) & (f <= fmax) & (coh2 >= coherence_min)
        if not np.any(band := mband):
            # 若全部被遮，放寬相干門檻一次（保留少量點以免完全空）
            mband = (f >= fmin) & (f <= fmax)
            band = mband

        # 解包相位並做加權線性去趨勢
        phi = np.unwrap(np.angle(Cxy))
        x = f[band]
        y = phi[band]
        w = coh2[band]
        if x.size < 8:  # 太少無法穩定估
            continue
        a0, a1 = _weighted_linfit(x, y, w)
        phi_res = y - (a0 + a1 * x)

        # 分 bin 聚合（log-space）
        edges = _logspace_edges(fmin, fmax, n_bins)
        k_list, y_list, w_list = [], [], []
        for i in range(n_bins):
            flo, fhi = edges[i], edges[i+1]
            sel = (x >= flo) & (x < fhi)
            if np.sum(sel) < max(3, int(0.5 * (nperseg // 512))):  # 每 bin 至少幾個點
                # 點太少則略過此 bin
                k_list.append(np.nan); y_list.append(np.nan); w_list.append(0.0); continue
            fmid = np.sqrt(flo * fhi)
            kbin = 2.0 * np.pi * fmid / C_LIGHT
            ybin = _bin_rms(np.abs(phi_res[sel]), w[sel])  # 以加權 RMS(|φ_res|) 作 y
            wbin = float(np.sum(w[sel]))                   # 權重：該 bin 的相干總量
            k_list.append(kbin); y_list.append(ybin); w_list.append(wbin)

        pair_bins[(a, b)] = {
            "k": np.array(k_list),
            "y": np.array(y_list),
            "w": np.array(w_list),
        }

    # 把 pair 結果聚合回每個 IFO（同一個 bin index 聚合該 IFO 參與到的所有 pair）
    edges = _logspace_edges(fmin, fmax, n_bins)
    rows = []
    for ifo in ifos:
        for ib in range(n_bins):
            ys, ws, ks = [], [], []
            for (a, b), dump in pair_bins.items():
                if ifo not in (a, b):
                    continue
                yb = dump["y"][ib]; wb = dump["w"][ib]; kb = dump["k"][ib]
                if not np.isfinite(yb) or wb <= 0.0 or not np.isfinite(kb):
                    continue
                ys.append(yb); ws.append(wb); ks.append(kb)
            if len(ys) == 0:
                continue
            # 加權平均（y 無單位 proxy，w 為該 bin 的相干總量）
            y_agg = float(np.sum(np.asarray(ws) * np.asarray(ys)) / np.sum(ws))
            # k：各 pair 同一 bin 其實相同，用平均以保險
            k_agg = float(np.mean(ks))
            w_agg = float(np.sum(ws))
            rows.append({"event": event, "ifo": ifo, "k": k_agg, "y": y_agg, "w": w_agg})

    # 處理過 sparse 的情況：確保每 IFO 至少有 min_bins_count 個 bin
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ok = df.groupby("ifo")["k"].count() >= min_bins_count
    keep_ifos = set(ok[ok].index.tolist())
    df = df[df["ifo"].isin(keep_ifos)].reset_index(drop=True)
    return df


# ------------------------
# 小工具：讀取 whitened 與 IFO 掃描
# ------------------------
def _detect_ifos(white_dir: Path, event: str) -> List[str]:
    ifos = []
    for p in sorted((white_dir).glob(f"{event}_*.npz")):
        name = p.stem  # e.g., GW170817_H1
        tag = name.split("_")[-1]
        ifos.append(tag)
    if not ifos:
        # 也接受 raw 目錄中 .npz 的命名
        for p in sorted((white_dir.parent.parent / "raw" / "gwosc" / event).glob("*.npz")):
            tag = p.stem  # H1.npz
            if len(tag) in (2, 3):  # H1/L1/V1
                ifos.append(tag)
    return sorted(list(set(ifos)))


def _load_whitened(work_dir: Path, event: str) -> Dict[str, Dict]:
    """
    回傳 {ifo: {"t": t, "w": whitened_series, "meta": meta_dict}}
    """
    d: Dict[str, Dict] = {}
    for p in sorted((work_dir).glob(f"{event}_*.npz")):
        with np.load(p, allow_pickle=True) as z:
            t = z["t"]; w = z["w"]
            # meta 在 earlier pipeline 是以 json.dumps(meta) 存
            meta = json.loads(str(z["meta"]))
        ifo = p.stem.split("_")[-1]
        d[ifo] = {"t": t, "w": w, "meta": meta}
    if not d:
        raise FileNotFoundError(f"no whitened npz found under {work_dir} for event={event}")
    return d
