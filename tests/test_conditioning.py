# tests/test_conditioning.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from uclgw.io.gwosc import synthesize_event
from uclgw.preprocess.conditioning import prepare_event

def test_prepare_event(tmp_path):
    raw = tmp_path / "raw/gwosc"
    seg = tmp_path / "work/segments"
    out = synthesize_event("GW170817", ["H1"], fs=2048.0, duration=16.0, seed=5,
                           raw_root=raw, seg_root=seg)
    seg_json = json.loads((seg/"GW170817.json").read_text())
    res = prepare_event(seg_json, psd_dir=tmp_path/"work/psd", white_dir=tmp_path/"work/whitened",
                        nperseg=2048, noverlap=1024, gate_z=5.0)
    # PSD file ok & positive
    psd_path = Path(res["psd"][0])
    df = pd.read_csv(psd_path)
    assert (df["psd"] > 0).all()
    # whitened series ok & roughly unit std
    wpath = Path(res["whitened"][0])
    dat = np.load(wpath, allow_pickle=True)
    w = dat["w"]
    assert w.size > 0
    assert 0.5 < float(np.std(w)) < 2.5

