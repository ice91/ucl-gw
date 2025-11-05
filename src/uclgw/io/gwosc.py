# src/uclgw/io/gwosc.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

# --------- 合成資料：維持既有行為，寫入 .npz ----------
def _make_chirp(n, fs, f0=30.0, f1=512.0):
    """非常簡化的 chirp（僅供煙霧測試）"""
    t = np.arange(n) / fs
    # 線性掃頻
    k = (f1 - f0) / t[-1]
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
    env = np.minimum(1.0, t / (0.1 + 1e-9)) * np.exp(-0.5 * (t / (t[-1] + 1e-9)))
    return env * np.sin(phase)

def synthesize_event(event: str, ifos: list[str], fs: float, duration: float, seed: int,
                     raw_root: Path, seg_root: Path):
    """離線合成波形，寫入 npz，並輸出 segments.json"""
    rng = np.random.default_rng(seed)
    n = int(fs * duration)

    out_segments = []
    evt_dir = raw_root / event
    evt_dir.mkdir(parents=True, exist_ok=True)

    for ifo in ifos:
        noise = rng.normal(0, 1e-21, size=n)
        h = noise + 5e-22 * _make_chirp(n, fs)
        t = np.arange(n, dtype=float) / fs  # 相對時間（秒）

        meta = {
            "event": event, "ifo": ifo, "fs": float(fs),
            "duration": float(duration), "source": "synthetic",
        }
        path = evt_dir / f"{ifo}.npz"
        np.savez_compressed(path, t=t, h=h, meta=json.dumps(meta))

        out_segments.append({
            "event": event,
            "ifo": ifo,
            "fs": float(fs),
            "path_raw": str(path),
            "gps_start": None,
            "gps_end": None,
            "source": "synthetic",
        })

    seg = {"event": event, "segments": out_segments}
    seg_root.mkdir(parents=True, exist_ok=True)
    with open(seg_root / f"{event}.json", "w") as f:
        json.dump(seg, f, indent=2)
    return seg

# --------- 真實資料：GWOSC 下載 .h5，寫入 segments.json ----------
def fetch_gwosc_event(event: str, ifos: list[str], fs: float, duration: float,
                      raw_root: Path, seg_root: Path):
    """
    以 GWOSC 開放資料下載事件附近的真實 strain：
    - 每個 IFO 取 event_gps ± duration/2
    - 重取樣至 sample_rate=fs
    - 儲存為 HDF5：data/raw/gwosc/{EVENT}/{IFO}.h5
    - 輸出 segments/{EVENT}.json（path_raw 指向 .h5）
    """
    try:
        from gwosc.datasets import event_gps
        from gwpy.timeseries import TimeSeries
    except Exception as e:
        raise SystemExit(
            "Missing gwpy/gwosc. Please install:\n"
            "  pip install gwpy gwosc h5py\n"
            f"Original import error: {e}"
        )

    t0 = float(event_gps(event))
    start = t0 - duration / 2.0
    end = t0 + duration / 2.0

    evt_dir = raw_root / event
    evt_dir.mkdir(parents=True, exist_ok=True)

    out_segments = []
    for ifo in ifos:
        # 下載 + 重取樣
        ts = TimeSeries.fetch_open_data(ifo, start, end, sample_rate=fs, cache=True)
        # 寫 HDF5（gwpy 自帶 HDF5 格式，可用 TimeSeries.read 讀回）
        h5_path = evt_dir / f"{ifo}.h5"
        ts.write(str(h5_path), format="hdf5")

        out_segments.append({
            "event": event,
            "ifo": ifo,
            "fs": float(ts.sample_rate.value),
            "path_raw": str(h5_path),
            "gps_start": float(ts.t0.value),
            "gps_end": float(ts.t0.value + ts.duration.value),
            "event_gps": t0,
            "duration": float(duration),
            "source": "gwosc",
            "url": f"https://www.gw-openscience.org/eventapi/html/event/{event}/"
        })

    seg = {"event": event, "gps": t0, "segments": out_segments}
    seg_root.mkdir(parents=True, exist_ok=True)
    with open(seg_root / f"{event}.json", "w") as f:
        json.dump(seg, f, indent=2)
    return seg
