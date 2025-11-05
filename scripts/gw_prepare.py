# scripts/gw_prepare.py
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from uclgw.preprocess.conditioning import prepare_event  # noqa: E402

def main():
    ap = argparse.ArgumentParser(description="Compute PSD (Welch) and whitened strain for each IFO segment.")
    ap.add_argument("--event", default="GW170817")
    ap.add_argument("--nperseg", type=int, default=4096*4)
    ap.add_argument("--noverlap", type=int, default=None)
    ap.add_argument("--gate-z", type=float, default=5.0, help="z-score gate threshold")
    args = ap.parse_args()

    seg_path = ROOT / f"data/work/segments/{args.event}.json"
    if not seg_path.exists():
        raise SystemExit(f"missing {seg_path}; run gw_fetch_gwosc.py first")
    with open(seg_path) as f:
        seg = json.load(f)

    out = prepare_event(
        seg, psd_dir=ROOT / "data/work/psd", white_dir=ROOT / "data/work/whitened",
        nperseg=args.nperseg, noverlap=args.noverlap, gate_z=args.gate_z
    )
    print(f"Prepared PSD and whitened for event {args.event}: {out}")

if __name__ == "__main__":
    main()
