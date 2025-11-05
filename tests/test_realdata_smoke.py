import os, json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.skipif(
    not bool(__import__("importlib").util.find_spec("gwpy")) or
    not bool(__import__("importlib").util.find_spec("gwosc")),
    reason="gwpy/gwosc not installed; skip real-data smoke."
)
def test_fetch_prepare_realdata_smoke(monkeypatch):
    from subprocess import run, CalledProcessError

    # 短一點的 duration/採樣率，減少下載量與時間
    env = os.environ.copy()
    cmd_fetch = [
        str(ROOT / ".venv/bin/python"),
        "-m", "scripts.gw_fetch_gwosc", "--event", "GW170817",
        "--ifos", "H1,L1", "--duration", "8", "--fs", "1024", "--download-gwosc"
    ]
    r = run(cmd_fetch, cwd=ROOT)
    assert r.returncode == 0

    # 準備（應產生 PSD 與 whitened）
    cmd_prep = [
        str(ROOT / ".venv/bin/python"),
        "-m", "scripts.gw_prepare", "--event", "GW170817",
        "--nperseg", "2048", "--noverlap", "1024", "--gate-z", "6.0"
    ]
    r2 = run(cmd_prep, cwd=ROOT)
    assert r2.returncode == 0

    seg_path = ROOT / "data/work/segments/GW170817.json"
    assert seg_path.exists(), "segments json not found"
    with open(seg_path) as f:
        seg = json.load(f)
    for s in seg["segments"]:
        assert s["path_raw"].endswith(".h5"), "expect real-data .h5"
