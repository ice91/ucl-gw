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
    相容老版欄位：若輸入含 y/w 或缺 f_hz，就補齊/改名成標準六欄。
    標準欄位: ['event','ifo','f_hz','k','delta_ct2','sigma']
    """
    df = df.copy()
    # f_hz
    if "f_hz" not in df.columns and "k" in df.columns:
        df["f_hz"] = (df["k"].astype(float) * C_LIGHT) / (2.0*np.pi)
    # delta_ct2
    if "delta_ct2" not in df.columns and "y" in df.columns:
        df["delta_ct2"] = df["y"].astype(float).clip(lower=1e-18)
    # sigma：等權（常數 1.0）
    if "sigma" not in df.columns:
        df["sigma"] = 1.0
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

    # 轉給 phasefit_points 的頻域/相干度/邊界丟棄參數
    ap.add_argument("--coh-min", type=float, default=0.7, help="Coherence^2 threshold for wideband trend & in-bin points")
    ap.add_argument("--coh-bin-min", type=float, default=0.80, help="Mean coherence^2 requirement inside each bin")
    ap.add_argument("--min-samples-per-bin", type=int, default=12, help="Minimum Welch samples per freq bin")
    ap.add_argument("--nperseg", type=int, default=8192, help="Welch FFT segment length")
    ap.add_argument("--noverlap", type=int, default=None, help="Welch overlap; default nperseg//2")
    ap.add_argument("--drop-edge-bins", type=int, default=0, help="Drop this many lowest & highest freq bins")

    args = ap.parse_args()

    # 加上 label 的外顯事件名（用於輸出檔案與 CSV 的 event 欄位）
    event = args.event if not args.label else f"{args.event}_{args.label}"

    # (A) 網路延遲側寫（JSON）— 仍以原始事件做計算，但輸出檔名帶 label，方便對照
    netrep = network_delay_bounds(event=args.event, work_dir=ROOT / "data/work/whitened")
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / f"reports/network_timing_{event}.json").write_text(json.dumps(netrep, indent=2))
    print(f"Wrote {ROOT}/reports/network_timing_{event}.json")

    # (B) 產出事件 csv
    if args.mode == "proxy-k2":
        df = proxy_k2_points(
            event=event,                      # ★ 重要：帶入帶 label 的事件名
            fmin=args.fmin, fmax=args.fmax, n_bins=args.n_bins,
            work_dir=ROOT / "data/work/whitened"
        )
    else:
        df = phasefit_points(
            event=event,                      # ★ 重要：帶入帶 label 的事件名
            fmin=args.fmin, fmax=args.fmax, n_bins=args.n_bins,
            work_dir=ROOT / "data/work/whitened",
            null_mode=args.null,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
            coherence_min=args.coh_min,
            coherence_bin_min=args.coh_bin_min,        # ★ 新增：bin 內平均 coh^2 門檻
            min_samples_per_bin=args.min_samples_per_bin,  # ★ 新增：bin 內最少樣本數
            drop_edge_bins=args.drop_edge_bins,
        )

    df = _ensure_standard_schema(df)

    out_dir = ROOT / "data/ct/events"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_event = out_dir / f"{event}_ct_bounds.csv"
    df.to_csv(out_event, index=False)
    print(f"Wrote {out_event} with {len(df)} rows")

    # (C) 彙整（注意：如為 OFF/null，建議不要併入主總表；視你的流程決定）
    if args.aggregate:
        summary = append_event_points(
            out_event,
            out_csv=ROOT / "data/ct/ct_bounds.csv",
            report_path=ROOT / "reports/aggregate_summary.json"
        )
        print(f"Updated {summary.out_csv} (events={summary.n_events}, rows={summary.n_rows})")

if __name__ == "__main__":
    main()
