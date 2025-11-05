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
        acc = Pxx if acc is None else (acc + Pxx)
        m += 1
        idx += step
    if m == 0:
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
    for i in bad:
        lo = max(0, i-10); hi = min(x.size, i+10)
        x[i] = np.median(x[lo:hi])
    return x


def _whiten_full_series(h: np.ndarray, fs: float, f_psd: np.ndarray, pxx: np.ndarray):
    n = h.size
    H = np.fft.rfft(h)
    f_bins = np.fft.rfftfreq(n, d=1.0/fs)
    pxx_interp = np.interp(f_bins, f_psd, pxx, left=pxx[0], right=pxx[-1])
    amp = np.sqrt(np.maximum(pxx_interp, 1e-40))
    W = H / amp
    w = np.fft.irfft(W, n=n)
    w = (w - np.mean(w)) / (np.std(w) + 1e-12)
    return w


def _load_raw_series(path: Path, fs_hint: float | None = None):
    """
    載入 .npz（合成）或 .h5（GWOSC 真實資料）並回傳 (t, h, fs, meta)
    - .npz: fields t, h, meta(json)
    - .h5:  由 gwpy 讀回 TimeSeries
    """
    suf = path.suffix.lower()
    if suf == ".npz":
        dat = np.load(path, allow_pickle=True)
        t = dat["t"].astype(float)
        h = dat["h"].astype(float)
        meta = {}
        if "meta" in dat.files:
            try:
                meta = json.loads(str(dat["meta"]))
            except Exception:
                meta = {}
        fs = float(meta.get("fs", 1.0 / np.mean(np.diff(t))))
        return t, h, fs, meta

    if suf in (".h5", ".hdf5"):
        try:
            from gwpy.timeseries import TimeSeries
        except Exception as e:
            raise SystemExit(
                "Reading .h5 requires gwpy. Install with: pip install gwpy h5py\n"
                f"Original import error: {e}"
            )
        ts = TimeSeries.read(str(path))
        h = ts.value.astype(float)
        fs = float(ts.sample_rate.value)
        # GPS 時間（秒）；這邊回傳絕對時間，後續僅用於保存。
        t = ts.times.value.astype(float)
        meta = {
            "fs": fs,
            "gps_start": float(ts.t0.value),
            "duration": float(ts.duration.value),
            "source": "gwosc"
        }
        return t, h, fs, meta

    raise SystemExit(f"Unsupported raw format: {path}")


def prepare_event(seg_json: dict, psd_dir: Path, white_dir: Path,
                  nperseg: int = 4096*4, noverlap: int | None = None,
                  gate_z: float = 5.0):
    psd_dir.mkdir(parents=True, exist_ok=True)
    white_dir.mkdir(parents=True, exist_ok=True)
    out = {"psd": [], "whitened": []}

    for s in seg_json["segments"]:
        raw_path = Path(s["path_raw"])
        # 優先使用 segment 提供的 meta（事件名/IFO/fs 等）
        meta = {
            "event": s.get("event", seg_json.get("event", "UNKNOWN")),
            "ifo": s.get("ifo", "NA"),
            "fs": float(s.get("fs", 0.0)),
            "gps_start": s.get("gps_start", None),
            "gps_end": s.get("gps_end", None),
            "source": s.get("source", "unknown")
        }

        t, h, fs_load, meta_load = _load_raw_series(raw_path, fs_hint=meta["fs"] or None)
        # 若 seg meta 缺欄位，用檔內 meta 補齊
        for k, v in meta_load.items():
            meta.setdefault(k, v)
        if not meta["fs"]:
            meta["fs"] = fs_load
        fs = float(meta["fs"])

        # gating -> PSD -> whiten
        h_g = _gate_outliers(h, zthr=gate_z)
        f_psd, pxx = _welch_psd(h_g, fs=fs, nperseg=nperseg, noverlap=noverlap)

        psd_path = psd_dir / f"{meta['event']}_{meta['ifo']}.csv"
        pd.DataFrame({"freq_hz": f_psd, "psd": pxx}).to_csv(psd_path, index=False)
        out["psd"].append(str(psd_path))

        w = _whiten_full_series(h_g, fs=fs, f_psd=f_psd, pxx=pxx)
        # 儲存相容的 whitened 介面（供下游共用）
        white_path = white_dir / f"{meta['event']}_{meta['ifo']}.npz"
        np.savez_compressed(white_path, t=t, w=w, meta=json.dumps(meta))
        out["whitened"].append(str(white_path))

    return out
