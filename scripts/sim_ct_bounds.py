# scripts/sim_ct_bounds.py
import sys, argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from uclgw.sim.synth import SynthConfig, synth_ct_bounds

# 盡量沿用你現有的 aggregate 工具（若存在）
try:
    from uclgw.combine.aggregate import append_event_points
except Exception:
    append_event_points = None

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic ct_bounds for Slope-2 validation")
    ap.add_argument("--event", default="SIM_S2")
    ap.add_argument("--ifos", default="H1,L1,V1")
    ap.add_argument("--fmin", type=float, default=80.0)
    ap.add_argument("--fmax", type=float, default=280.0)
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--slope", type=float, default=2.0)
    ap.add_argument("--log10A", type=float, default=-10.0)
    ap.add_argument("--sigma-rel", type=float, default=0.10)
    ap.add_argument("--hetero", type=int, default=1)
    ap.add_argument("--p-out", type=float, default=0.0)
    ap.add_argument("--out-mult", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--append", action="store_true", help="Append to data/ct/ct_bounds.csv via aggregate")
    args = ap.parse_args()

    cfg = SynthConfig(
        event=args.event,
        ifos=[x.strip() for x in args.ifos.split(",") if x.strip()],
        fmin=args.fmin, fmax=args.fmax, n_bins=args.n_bins,
        slope=args.slope, log10A=args.log10A,
        sigma_rel=args.sigma_rel, hetero=bool(args.hetero),
        p_out=args.p_out, out_mult=args.out_mult, seed=args.seed
    )
    df = synth_ct_bounds(cfg)

    out_dir = ROOT / "data" / "ct" / "events"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{cfg.event}_ct_bounds.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows")

    if args.append:
        if append_event_points is None:
            print("[WARN] append_event_points not available; skip aggregate.")
        else:
            summary = append_event_points(
                out_csv,
                out_csv=ROOT / "data/ct/ct_bounds.csv",
                report_path=ROOT / "reports/aggregate_summary.json"
            )
            print(f"Updated {summary.out_csv} (events={summary.n_events}, rows={summary.n_rows})")

if __name__ == "__main__":
    main()
