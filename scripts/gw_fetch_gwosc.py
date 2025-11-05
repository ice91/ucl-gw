# scripts/gw_fetch_gwosc.py
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from uclgw.io.gwosc import synthesize_event, fetch_gwosc_event  # noqa: E402


def parse_ifos(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Fetch/Index GWOSC event segments. "
                    "Default: synthetic offline; real data via --download-gwosc."
    )
    ap.add_argument("--event", default="GW170817")
    ap.add_argument("--ifos", default="H1,L1,V1")
    ap.add_argument("--duration", type=float, default=32.0)
    ap.add_argument("--fs", type=float, default=4096.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--download-gwosc", action="store_true",
                    help="Download real open data from GWOSC into data/raw/gwosc/{EVENT}/*.h5")
    args = ap.parse_args()

    ifos = parse_ifos(args.ifos)
    raw_root = ROOT / "data/raw/gwosc"
    seg_root = ROOT / "data/work/segments"
    seg_root.mkdir(parents=True, exist_ok=True)

    if args.download_gwosc:
        out = fetch_gwosc_event(
            event=args.event, ifos=ifos, fs=args.fs, duration=args.duration,
            raw_root=raw_root, seg_root=seg_root,
        )
    else:
        out = synthesize_event(
            event=args.event, ifos=ifos, fs=args.fs, duration=args.duration, seed=args.seed,
            raw_root=raw_root, seg_root=seg_root
        )

    seg_path = seg_root / f"{args.event}.json"
    print(f"Wrote segments: {seg_path} with {len(out['segments'])} entries")


if __name__ == "__main__":
    main()
