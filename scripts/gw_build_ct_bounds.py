# scripts/gw_build_ct_bounds.py
# scripts/gw_build_ct_bounds.py（最上方匯入段落）
from src.uclgw.features.network_timing import estimate_delays_and_bounds as rough_network_timing
from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd

from src.uclgw.features.network_timing import rough_network_timing
from src.uclgw.features.dispersion_fit import make_proxy_k2_points  # Phase 2 已有
from src.uclgw.combine.aggregate import aggregate

def write_event_csv(event: str, rows: list[dict], out_dir: str = "data/ct/events") -> str:
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{event}_ct_bounds.csv")
    pd.DataFrame(rows, columns=["event", "ifo", "f_hz", "k", "delta_ct2", "sigma"]).to_csv(out, index=False)
    return out

def main():
    ap = argparse.ArgumentParser(description="Build per-event ct_bounds and optionally aggregate all events.")
    ap.add_argument("--event", required=True, help="Event name, e.g., GW170817")
    ap.add_argument("--fmin", type=float, default=30.0)
    ap.add_argument("--fmax", type=float, default=1024.0)
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--mode", type=str, default="proxy-k2",
                    choices=["proxy-k2"],  # 後續可擴充: ppe-residual
                    help="How to generate delta_ct2 vs k points")
    ap.add_argument("--aggregate", action="store_true", help="Also aggregate into data/ct/ct_bounds.csv")
    ap.add_argument("--segments", default="data/work/segments", help="segments JSON folder")
    ap.add_argument("--whitened", default="data/work/whitened", help="whitened npz folder")
    args = ap.parse_args()

    # 1) network timing sanity（寫報告讓審稿人看到）
    timing = rough_network_timing(event=args.event, segments_dir=args.segments)
    rep_path = os.path.join("reports", f"network_timing_{args.event}.json")
    os.makedirs("reports", exist_ok=True)
    with open(rep_path, "w") as f:
        json.dump(timing, f, indent=2)
    print(f"Wrote {os.path.abspath(rep_path)}")

    # 2) dispersion / proxy 生成 per-event 離散點
    rows = make_proxy_k2_points(event=args.event,
                                whitened_dir=args.whitened,
                                fmin=args.fmin, fmax=args.fmax, n_bins=args.n_bins)
    event_csv = write_event_csv(args.event, rows)
    print(f"Wrote {os.path.abspath(event_csv)} with {len(rows)} rows")

    # 3) 合併成 data/ct/ct_bounds.csv（若指定）
    if args.aggregate:
        summary = aggregate(events_dir="data/ct/events",
                            out_csv="data/ct/ct_bounds.csv",
                            write_summary_json="reports/aggregate_summary.json")
        print(f"Updated {os.path.abspath('data/ct/ct_bounds.csv')} "
              f"(events={summary.n_events}, rows={summary.n_rows})")

if __name__ == "__main__":
    main()
