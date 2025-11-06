# scripts/slope2_perifo.py
# --- add project src to path ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
# --------------------------------

import argparse, json, numpy as np
import pandas as pd
from uclgw.eval.slopefit import load_ct, do_fit, _read_profile_window

EPS_FLOOR = 1e-14

def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 防 OFF/NULL 出現 0/負值導致 log 無效
    if "delta_ct2" in df.columns:
        df["delta_ct2"] = np.abs(df["delta_ct2"]).clip(lower=EPS_FLOOR)
    return df

def _finite_in_window(df: pd.DataFrame, kmin: float, kmax: float) -> pd.DataFrame:
    m = np.isfinite(df["k"].values) & np.isfinite(df["delta_ct2"].values)
    m &= (df["k"].values >= kmin) & (df["k"].values <= kmax)
    return df[m].copy()

def main():
    ap = argparse.ArgumentParser(description="Per-IFO slope2 fit per event.")
    ap.add_argument("--data", default="data/ct/ct_bounds.csv")
    ap.add_argument("--profile", default="configs/profiles/lvk_o3.yaml")
    ap.add_argument("--event", default=None)
    ap.add_argument("--method", default="wls", choices=["wls","huber"])
    args = ap.parse_args()

    kmin, kmax = _read_profile_window(Path(args.profile))
    df = load_ct(Path(args.data))
    if args.event:
        df = df[df["event"] == args.event]
    if df.empty:
        print("No rows for the selected event/data.")
        return

    df = _sanitize_df(df)
    df = _finite_in_window(df, kmin, kmax)
    if df.empty:
        print("No valid rows after finite filtering.")
        return

    events = sorted(df["event"].unique())
    by_event = {}

    for ev in events:
        dfe = df[df["event"] == ev].copy()
        if dfe.empty: continue

        # 合併（全 IFO）
        tmp_all = Path("data/ct/_tmp_real.csv")
        dfe.to_csv(tmp_all, index=False)
        r_all = do_fit(tmp_all, Path(args.profile), method=args.method)

        perifo = {}
        for ifo in sorted(dfe["ifo"].unique()):
            dfi = dfe[dfe["ifo"] == ifo].copy()
            if dfi.empty: continue
            tmp_1 = Path("data/ct/_tmp_real.csv")
            dfi.to_csv(tmp_1, index=False)
            r = do_fit(tmp_1, Path(args.profile), method=args.method)
            perifo[ifo] = {
                "slope": float(r.slope),
                "intercept_log10": float(r.intercept),
                "n": int(getattr(r, "mask_count", len(dfi))),
                "method": args.method,
                "window_k": [float(kmin), float(kmax)]
            }

        by_event[ev] = {
            "per_ifo": perifo,
            "combined": {
                "slope": float(r_all.slope),
                "intercept_log10": float(r_all.intercept),
                "n": int(getattr(r_all, "mask_count", len(dfe))),
                "method": args.method,
                "window_k": [float(kmin), float(kmax)]
            }
        }

    out = {"by_event": by_event, "overall": {"n_events": len(by_event)}}
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/slope2_perifo.json").write_text(json.dumps(out, indent=2))
    print("Wrote reports/slope2_perifo.json")

if __name__ == "__main__":
    main()
