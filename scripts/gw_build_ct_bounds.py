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

C_LIGHT = 299_792_458.0

def _ensure_standard_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    保障輸出欄位統一為：
      ['event','ifo','f_hz','k','delta_ct2','sigma']
    - 雙向補欄：缺 f_hz 用 k 還原；缺 k 用 f_hz 推回
    - 數值轉型；delta_ct2 加下界夾制
    - sigma 缺省補 1.0
    """
    required = ["event","ifo","f_hz","k","delta_ct2","sigma"]
    df = df.copy()
    if df.empty:
        return pd.DataFrame(columns=required)

    # 先把已存在欄位轉成數值以避免 obj/str 造成 NaN
    for c in ["f_hz","k","delta_ct2","sigma"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 雙向補齊 f_hz / k
    if "f_hz" not in df.columns and "k" in df.columns:
        df["f_hz"] = (df["k"].astype(float) * C_LIGHT) / (2.0*np.pi)
    if "k" not in df.columns and "f_hz" in df.columns:
        df["k"] = (2.0*np.pi*df["f_hz"].astype(float)) / C_LIGHT

    # 其他欄位保險
    if "delta_ct2" not in df.columns and "y" in df.columns:
        df["delta_ct2"] = pd.to_numeric(df["y"], errors="coerce")
    if "sigma" not in df.columns:
        df["sigma"] = 1.0

    # 物理下界（避免非正數進 log）
    if "delta_ct2" in df.columns:
        df["delta_ct2"] = df["delta_ct2"].astype(float).clip(lower=1e-18)

    # 最終欄序
    out = pd.DataFrame(columns=required)
    for c in required:
        if c in df.columns:
            out[c] = df[c]
    # event/ifo 轉字串以避免之後 concat 出現混型
    if "event" in out.columns:
        out["event"] = out["event"].astype(str)
    if "ifo" in out.columns:
        out["ifo"] = out["ifo"].astype(str)
    return out[required]

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

    # phase-fit 參數
    ap.add_argument("--coh-min", type=float, default=0.7, help="coherence^2 threshold for per-bin local fit")
    ap.add_argument("--coh-wide-min", type=float, default=0.80, help="coherence^2 threshold used ONLY for the wideband trend fit (L_eff)")
    ap.add_argument("--coh-bin-min", type=float, default=0.80, help="mean coherence^2 requirement inside each bin")
    ap.add_argument("--min-samples-per-bin", type=int, default=12, help="minimum Welch samples per frequency bin")
    ap.add_argument("--nperseg", type=int, default=8192, help="Welch FFT segment length")
    ap.add_argument("--noverlap", type=int, default=None, help="Welch overlap; default nperseg//2")
    ap.add_argument("--drop-edge-bins", type=int, default=0, help="Drop this many lowest & highest freq bins")
    ap.add_argument("--min-bins-count", type=int, default=6, help="minimum surviving bins per IFO to keep that IFO")
    ap.add_argument("--gate-sec", type=float, default=0.0, help="half-width (seconds) of event-centered time gate; 0 disables")
    ap.add_argument("--edges-mode", choices=["adaptive", "logspace"], default="adaptive")
    ap.add_argument("--coh-min-edges", type=float, default=None, help="only for adaptive edges; default = use coh-min")

    args = ap.parse_args()

    source_event = args.event
    event = args.event if not args.label else f"{args.event}_{args.label}"

    # 填補 edges 門檻預設
    if args.coh_min_edges is None:
        args.coh_min_edges = args.coh_min

    netrep = network_delay_bounds(event=source_event, work_dir=ROOT / "data/work/whitened")
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / f"reports/network_timing_{event}.json").write_text(json.dumps(netrep, indent=2))
    print(f"Wrote {ROOT}/reports/network_timing_{event}.json")

    if args.mode == "proxy-k2":
        df = proxy_k2_points(
            event=event,
            fmin=args.fmin, fmax=args.fmax, n_bins=args.n_bins,
            work_dir=ROOT / "data/work/whitened"
        )
    else:
        df = phasefit_points(
            event=event,
            source_event=source_event,
            fmin=args.fmin, fmax=args.fmax, n_bins=args.n_bins,
            work_dir=ROOT / "data/work/whitened",
            null_mode=args.null,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
            coherence_min=args.coh_min,
            coherence_wide_min=args.coh_wide_min,
            coherence_bin_min=args.coh_bin_min,
            min_samples_per_bin=args.min_samples_per_bin,
            drop_edge_bins=args.drop_edge_bins,
            min_bins_count=args.min_bins_count,
            gate_sec=args.gate_sec,
            edges_mode=args.edges_mode,
            coherence_edges_min=args.coh_min_edges,
        )

    df = _ensure_standard_schema(df)

    out_dir = ROOT / "data/ct/events"; out_dir.mkdir(parents=True, exist_ok=True)
    out_event = out_dir / f"{event}_ct_bounds.csv"
    df.to_csv(out_event, index=False)
    print(f"Wrote {out_event} with {len(df)} rows")

    if args.aggregate:
        if df.empty:
            print("No bins survived; skip aggregate to avoid empty-event issues.")
        else:
            summary = append_event_points(
                out_event,
                out_csv=ROOT / "data/ct/ct_bounds.csv",
                report_path=ROOT / "reports/aggregate_summary.json"
            )
            print(f"Updated {summary.out_csv} (events={summary.n_events}, rows={summary.n_rows})")

if __name__ == "__main__":
    main()
