# scripts/gw_build_ct_bounds.py
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from uclgw.features.network_timing import network_delay_bounds
from uclgw.features.dispersion_fit import proxy_k2_points, phasefit_points
from uclgw.combine.aggregate import append_event_points

C_LIGHT = 299792458.0

def _ensure_standard_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    兼容：若輸入含有 y/w 或缺 f_hz，就補齊/改名成標準六欄。
    標準欄位: ['event','ifo','f_hz','k','delta_ct2','sigma']
    """
    df = df.copy()
    # f_hz
    if "f_hz" not in df.columns and "k" in df.columns:
        df["f_hz"] = (df["k"].astype(float) * C_LIGHT) / (2.0*np.pi)
    # delta_ct2
    if "delta_ct2" not in df.columns and "y" in df.columns:
        df["delta_ct2"] = df["y"].astype(float).clip(lower=1e-18)
    # sigma
    if "sigma" not in df.columns and "w" in df.columns:
        w = df["w"].astype(float).clip(lower=0.0)
        d = df["delta_ct2"].astype(float).clip(lower=1e-18)
        df["sigma"] = d / np.sqrt(w + 1e-9)
    # 最終只保留標準欄位（若還缺就讓上游報錯）
    keep = ["event","ifo","f_hz","k","delta_ct2","sigma"]
    return df[keep]

def main():
    ap = argparse.ArgumentParser(description="Build δc_T^2(k) points for an event (and optionally aggregate).")
    ap.add_argument("--event", default="GW170817")
    ap.add_argument("--fmin", type=float, default=30.0)
    ap.add_argument("--fmax", type=float, default=1024.0)
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--mode", choices=["proxy-k2", "phase-fit"], default="proxy-k2")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--null", choices=["none", "timeshift"], default="none", help="Apply a laboratory null.")
    ap.add_argument("--label", default="", help="Suffix for event name (e.g. OFF)")
    args = ap.parse_args()

    event = args.event if not args.label else f"{args.event}_{args.label}"

    # (A) 網路延遲側寫（JSON）
    netrep = network_delay_bounds(event=args.event, work_dir=ROOT / "data/work/whitened")
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / f"reports/network_timing_{event}.json").write_text(json.dumps(netrep, indent=2))
    print(f"Wrote {ROOT}/reports/network_timing_{event}.json")

    # (B) 產出事件 csv
    if args.mode == "proxy-k2":
        df = proxy_k2_points(event=args.event, fmin=args.fmin, fmax=args.fmax, n_bins=args.n_bins,
                             work_dir=ROOT / "data/work/whitened")
    else:
        df = phasefit_points(event=args.event, fmin=args.fmin, fmax=args.fmax, n_bins=args.n_bins,
                             work_dir=ROOT / "data/work/whitened", null_mode=args.null)

    df = _ensure_standard_schema(df)

    out_dir = ROOT / "data/ct/events"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_event = out_dir / f"{event}_ct_bounds.csv"
    df.to_csv(out_event, index=False)
    print(f"Wrote {out_event} with {len(df)} rows")

    # (C) 彙整
    if args.aggregate:
        summary = append_event_points(out_event,
                                      out_csv=ROOT / "data/ct/ct_bounds.csv",
                                      report_path=ROOT / "reports/aggregate_summary.json")
        print(f"Updated {summary.out_csv} (events={summary.n_events}, rows={summary.n_rows})")

if __name__ == "__main__":
    main()
