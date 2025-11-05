# tests/test_gwosc_io.py
import json
from pathlib import Path
from uclgw.io.gwosc import synthesize_event

def test_synthesize_event(tmp_path):
    raw = tmp_path / "raw/gwosc"
    seg = tmp_path / "work/segments"
    out = synthesize_event("GW170817", ["H1","L1"], fs=2048.0, duration=16.0, seed=3,
                           raw_root=raw, seg_root=seg)
    segf = seg / "GW170817.json"
    assert segf.exists()
    j = json.loads(segf.read_text())
    assert j["event"] == "GW170817"
    for s in j["segments"]:
        assert Path(s["path_raw"]).exists()
        assert s["n"] > 0
