# scripts/gw_build_ct_bounds.py
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from uclgw.features.network_timing import estimate_delays_and_bounds  # noqa: E402
from uclgw.features.dispersion_fit import build_ct_points_from_whitened  # noqa: E402
from uclgw.combine.aggregate import append_event_to_master  # noqa: E402

def main():
    ap = argparse.ArgumentParser(description="Build δc_T^2(k) event table and aggregate master CSV.")
    ap.add_argument("--event", default="GW170817")
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=1024.0)
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--mode", default="proxy-k2", choices=["proxy-k2"])
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    white_dir = ROOT / "data/work/whitened"
    out_dir_ev = ROOT / "data/ct/events"
    out_dir_ev.mkdir(parents=True, exist_ok=True)
    out_dir_ct = ROOT / "data/ct"
    out_dir_ct.mkdir(parents=True, exist_ok=True)

    # collect whitened paths for the event
    white_paths = sorted(white_dir.glob(f"{args.event}_*.npz"))
    if not white_paths:
        raise SystemExit(f"No whitened files for event {args.event} in {white_dir}; run make gw-prepare first.")

    # 1) network timing sanity → reports/network_timing_{EVENT}.json
    nt = estimate_delays_and_bounds(white_paths)
    rep_path = ROOT / f"reports/network_timing_{args.event}.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(nt, indent=2))
    print(f"Wrote {rep_path}")

    # 2) dispersion-based δc_T^2(k) points → event csv
    df = build_ct_points_from_whitened(white_paths, fmin=args.fmin, fmax=args.fmax,
                                       n_bins=args.n_bins, mode=args.mode)
    ev_csv = out_dir_ev / f"{args.event}_ct_bounds.csv"
    df.to_csv(ev_csv, index=False)
    print(f"Wrote {ev_csv} with {len(df)} rows")

    # 3) aggregate
    if args.aggregate:
        master = out_dir_ct / "ct_bounds.csv"
        append_event_to_master(ev_csv, master)
        print(f"Updated {master}")

if __name__ == "__main__":
    main()

