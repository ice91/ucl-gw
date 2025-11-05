# scripts/gw_fetch_gwosc.py
import argparse, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from uclgw.io.gwosc import synthesize_event  # noqa: E402

def parse_ifos(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser(description="Fetch/Index GWOSC event segments (offline synth for Phase 1).")
    ap.add_argument("--event", default="GW170817")
    ap.add_argument("--ifos", default="H1,L1,V1")
    ap.add_argument("--duration", type=float, default=32.0)
    ap.add_argument("--fs", type=float, default=4096.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ifos = parse_ifos(args.ifos)
    out = synthesize_event(
        event=args.event, ifos=ifos, fs=args.fs, duration=args.duration, seed=args.seed,
        raw_root=ROOT / "data/raw/gwosc", seg_root=ROOT / "data/work/segments"
    )
    seg_path = ROOT / f"data/work/segments/{args.event}.json"
    print(f"Wrote segments: {seg_path} with {len(out['segments'])} entries")

if __name__ == "__main__":
    main()
