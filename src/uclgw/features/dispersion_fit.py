# src/uclgw/features/dispersion_fit.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

C_LIGHT = 299_792_458.0
TWOPI = 2.0 * np.pi

# ------------------------
# 工具：Welch 版 CSD/Coherence
# ------------------------
def _welch_csd_xy(x: np.ndarray, y: np.ndarray, fs: float,
                  nperseg: int = 4096, noverlap: int | None = None):
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
    """
    權重線性回歸：y ≈ a0 + a1 x；回傳 (a0, a1)
    """
    w = np.asarray(w); x = np.asarray(x); y = np.asarray(y)
    W = np.sqrt(np.maximum(w, 1e-16))
    A = np.stack([np.ones_like(x), x], axis=1)
    Aw = A * W[:, None]; yw = y * W
    beta, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    a0, a1 = beta[0], beta[1]
    return float(a0), float(a1)


def _logspace_edges(fmin: float, fmax: float, n_bins: int) -> np.ndarray:
    return np.logspace(np.log10(fmin), np.log10(fmax), n_bins + 1)


def _adaptive_edges(f: np.ndarray, coh2: np.ndarray,
                    fmin: float, fmax: float, n_bins: int, coh_min: float) -> np.ndarray:
    """
    依據通過 coh_min 門檻的頻點，做「等分位」分箱；若點數不足則退回 logspace。
    """
    mask = (f >= fmin) & (f <= fmax) & (coh2 >= coh_min)
    f_ok = np.sort(f[mask])
    if f_ok.size < (n_bins + 1):
        return _logspace_edges(fmin, fmax, n_bins)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(f_ok, qs)
    # 保證嚴格遞增（遇到重複時微擾）
    for i in range(1, edges.size):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-6
    return edges


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
            k = TWOPI * fmid / C_LIGHT
            delta_ct2 = (k**2)  # 期望斜率=2 的代理量
            rows.append({
                "event": event, "ifo": ifo, "f_hz": fmid, "k": k,
                "delta_ct2": float(delta_ct2), "sigma": 1.0  # 等權
            })
    return pd.DataFrame(rows)

# ------------------------
# 真實：phase-fit
# ------------------------
def phasefit_points(
    event: str,                    # 輸出標籤（寫進 CSV 的 event 欄）
    fmin: float,
    fmax: float,
    n_bins: int,
    work_dir: Path,
    null_mode: str = "none",
    nperseg: int = 8192,
    noverlap: int | None = None,
    coherence_min: float = 0.7,
    coherence_wide_min: float = 0.8,
    coherence_bin_min: float = 0.80,
    min_samples_per_bin: int = 12,
    min_bins_count: int = 6,
    drop_edge_bins: int = 0,
    gate_sec: float = 0.0,
    source_event: Optional[str] = None,   # NEW：讀檔來源事件（預設沿用 event）
) -> pd.DataFrame:
    """
    讀檔用 source_event；輸出標籤用 event。
    """
    src_evt = source_event or event

    # 載入 whitened（以來源事件名查檔）
    W = _load_whitened(work_dir, src_evt)
    ifos = sorted(W.keys())
    if len(ifos) < 2:
        raise RuntimeError(f"need >=2 IFOs, got {ifos}")

    # 檢查取樣率一致
    fs_ref = None
    for ifo in ifos:
        fs = float(W[ifo]["meta"]["fs"])
        if fs_ref is None: fs_ref = fs
        if abs(fs - fs_ref) > 1e-9:
            raise RuntimeError("sampling rate mismatch across IFOs")

    # 事件窗 gating（使用所有 IFO 的 t0_idx 做「交集」視窗，確保對齊）
    if gate_sec and gate_sec > 0:
        t0_list = []
        for ifo in ifos:
            w = np.asarray(W[ifo]["w"])
            t0_idx = int(W[ifo]["meta"].get("t0_idx", int(np.argmax(np.abs(w)))))
            t0_list.append(t0_idx)
        rad = int(max(1, gate_sec * fs_ref))
        lo_cands = []
        hi_cands = []
        for ifo in ifos:
            w = np.asarray(W[ifo]["w"])
            t0_idx = int(W[ifo]["meta"].get("t0_idx", int(np.argmax(np.abs(w)))))
            lo_cands.append(max(0, t0_idx - rad))
            hi_cands.append(min(w.size, t0_idx + rad))
        lo = int(max(lo_cands)); hi = int(min(hi_cands))
        # 需保證切出來的長度足以做 Welch
        if hi - lo >= max(nperseg, 16):
            for ifo in ifos:
                w = np.asarray(W[ifo]["w"]); t = np.asarray(W[ifo]["t"])
                W[ifo]["w"] = w[lo:hi]
                W[ifo]["t"] = t[lo:hi]
        # 若交集過窄就維持原樣（不 gating）

    pairs = [(ifos[i], ifos[j]) for i in range(len(ifos)) for j in range(i+1, len(ifos))]
    f_ref = float(np.sqrt(fmin * fmax))  # 權重用參考頻率（幾何平均）

    # 先用第一個 pair 估出自適應分箱邊界（若失敗會自動退回 logspace）
    a0_ifo, b0_ifo = pairs[0]
    xa0 = W[a0_ifo]["w"].copy()
    yb0 = W[b0_ifo]["w"].copy()
    if null_mode == "timeshift":
        yb0 = _timeshift(yb0, shift_samples=(len(yb0) // 4))
    f0, Cxy0, Sxx0, Syy0 = _welch_csd_xy(xa0, yb0, fs=fs_ref, nperseg=nperseg, noverlap=noverlap)
    coh20 = np.clip((np.abs(Cxy0)**2) / (np.maximum(Sxx0,1e-30)*np.maximum(Syy0,1e-30)), 0.0, 1.0)
    edges = _adaptive_edges(f0, coh20, fmin, fmax, n_bins, coherence_min)

    pair_bins: Dict[tuple[str, str], Dict[str, np.ndarray]] = {}

    for (a, b) in pairs:
        xa = W[a]["w"].copy()
        yb = W[b]["w"].copy()
        if null_mode == "timeshift":
            yb = _timeshift(yb, shift_samples=(len(yb) // 4))

        f, Cxy, Sxx, Syy = _welch_csd_xy(xa, yb, fs=fs_ref, nperseg=nperseg, noverlap=noverlap)
        coh2 = np.clip((np.abs(Cxy)**2) / (np.maximum(Sxx,1e-30)*np.maximum(Syy,1e-30)), 0.0, 1.0)
        phi = np.unwrap(np.angle(Cxy))

        # (1) wideband 趨勢：只用較高相干點
        mband = (f >= fmin) & (f <= fmax) & (coh2 >= coherence_wide_min)
        if not np.any(mband):
            # 若沒有任何點達到 wideband 門檻，退而求其次用整段（避免 L_eff 無法估）
            mband = (f >= fmin) & (f <= fmax)
        a0, a1 = _weighted_linfit(f[mband], phi[mband], coh2[mband] if np.any(mband) else np.ones_like(f[mband]))
        dt_geom = a1 / TWOPI                         # [s]
        L_eff   = C_LIGHT * abs(dt_geom)             # [m]

        # (2) 去趨勢殘差
        phi_res = phi - (a0 + a1 * f)

        # (3) 逐 bin 估 b_loc → δc_T^2
        k_list, y_list, w_list, fmid_list = [], [], [], []
        for i in range(n_bins):
            # 丟掉邊緣 bin（避免窗函數邊效應）
            if drop_edge_bins > 0 and (i < drop_edge_bins or i >= (n_bins - drop_edge_bins)):
                k_list.append(np.nan); y_list.append(np.nan); w_list.append(0.0); fmid_list.append(np.nan); continue

            flo, fhi = float(edges[i]), float(edges[i+1])
            sel = (f >= flo) & (f < fhi) & (coh2 >= coherence_min)
            # 嚴格門檻：bin 內樣本數 + 平均 coh^2
            if np.sum(sel) < max(int(min_samples_per_bin), 1):
                k_list.append(np.nan); y_list.append(np.nan); w_list.append(0.0); fmid_list.append(np.nan); continue
            if float(np.mean(coh2[sel])) < float(coherence_bin_min):
                k_list.append(np.nan); y_list.append(np.nan); w_list.append(0.0); fmid_list.append(np.nan); continue

            a_loc, b_loc = _weighted_linfit(f[sel], phi_res[sel], coh2[sel])  # b_loc ≈ dφ/df (rad/Hz)
            fmid = float(np.sqrt(flo * fhi))
            kbin = float(TWOPI * fmid / C_LIGHT)

            # 物理解釋（校正 ×2；最後才取絕對值，避免正負互相抵銷）
            delta_ct2 = float((2.0 * C_LIGHT / (3.0 * np.pi * max(L_eff, 1e-9))) * abs(b_loc))

            # 權重：Σ coh^2 × (fmid/f_ref)^2 × L_eff^2（讓長基線與高頻更穩健）
            wbin = float(np.sum(coh2[sel])) * float((fmid / max(f_ref, 1e-9))**2) * float(L_eff**2)

            k_list.append(kbin); y_list.append(delta_ct2); w_list.append(wbin); fmid_list.append(fmid)

        pair_bins[(a,b)] = {
            "k": np.array(k_list), "y": np.array(y_list),
            "w": np.array(w_list), "fmid": np.array(fmid_list),
            "L_eff": float(L_eff)
        }

    # (4) IFO 聚合（權重加權平均），並把總權重轉成 sigma（軟降權）
    rows = []
    for ifo in ifos:
        for ib in range(n_bins):
            ys, ws, ks, fs = [], [], [], []
            for (a,b), dump in pair_bins.items():
                if ifo not in (a,b): continue
                yb = dump["y"][ib]; wb = dump["w"][ib]; kb = dump["k"][ib]; fm = dump["fmid"][ib]
                if not np.isfinite(yb) or wb <= 0.0 or not np.isfinite(kb) or not np.isfinite(fm):
                    continue
                ys.append(yb); ws.append(wb); ks.append(kb); fs.append(fm)
            if len(ys) == 0:
                continue

            ys = np.asarray(ys); ws = np.asarray(ws); ks = np.asarray(ks); fs = np.asarray(fs)
            wsum = float(np.sum(ws))
            y_agg = float(np.sum(ws*ys) / wsum)
            k_agg = float(np.sum(ws*ks) / wsum)
            f_agg = float(np.sum(ws*fs) / wsum)

            # sigma 以總權重作軟降權；避免過小，加入保險下限
            sigma = 1.0 / max(np.sqrt(wsum), 1e-3)

            rows.append({
                "event": event, "ifo": ifo, "f_hz": f_agg, "k": k_agg,
                "delta_ct2": max(y_agg, 1e-18), "sigma": float(sigma)
            })

    df = pd.DataFrame(rows, columns=["event","ifo","f_hz","k","delta_ct2","sigma"])
    if df.empty:
        return df
    ok = df.groupby("ifo")["k"].count() >= max(int(min_bins_count), 1)
    keep = set(ok[ok].index.tolist())
    df = df[df["ifo"].isin(keep)].reset_index(drop=True)
    return df