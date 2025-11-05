# src/uclgw/io/gwosc.py
from __future__ import annotations
import json, os
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

@dataclass
class Segment:
    ifo: str
    fs: float
    gps_start: float
    gps_end: float
    n: int
    path_raw: str
    channel: str = "STRAIN"

def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _sine_chirp(t, f0=30.0, f1=256.0, T=32.0, phi0=0.0):
    # linear chirp phase: phi(t) = 2π ( f0 t + 0.5 a t^2 ), a=(f1-f0)/T
    a = (f1 - f0) / max(1e-9, T)
    return np.sin(2*np.pi*(f0*t + 0.5*a*t**2) + phi0)

def synthesize_event(event: str, ifos, fs: float, duration: float, seed: int,
                     raw_root: Path, seg_root: Path):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, 1.0/fs)
    n = t.size
    gps0 = 1187008882.0  # arbitrary fixed base for GW170817-like
    segments = []

    event_dir = Path(raw_root) / event
    seg_dir = Path(seg_root)
    event_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    for i, ifo in enumerate(ifos):
        # noise + chirp
        noise = rng.normal(0.0, 1.0, size=n) * 1e-21
        chirp = _sine_chirp(t, f0=30, f1=280, T=duration, phi0=0.3*i) * 5e-21
        h = noise + chirp
        meta = {
            "event": event, "ifo": ifo, "fs": fs, "duration": duration,
            "gps_start": gps0, "gps_end": gps0 + duration, "channel": "STRAIN",
            "n": int(n)
        }
        raw_path = event_dir / f"{ifo}.npz"
        _ensure_dir(raw_path)
        np.savez_compressed(raw_path, t=t, h=h, meta=json.dumps(meta))
        segments.append(Segment(ifo=ifo, fs=fs, gps_start=gps0,
                                gps_end=gps0+duration, n=int(n),
                                path_raw=str(raw_path)))

    seg_json = {
        "event": event,
        "segments": [asdict(s) for s in segments]
    }
    seg_path = seg_dir / f"{event}.json"
    with open(seg_path, "w") as f:
        json.dump(seg_json, f, indent=2)
    return seg_json
