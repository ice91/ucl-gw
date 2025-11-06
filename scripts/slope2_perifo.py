# scripts/slope2_perifo.py
import sys, argparse, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import pandas as pd
from uclgw.eval.slopefit import load_ct, do_fit

def _get_intercept_any(r):
    if hasattr(r, "intercept_log10"): return float(r.intercept_log10)
    if hasattr(r, "intercept"): return float(r.intercept)
    return None

def _get_n_used(r, fallback_len):
    for attr in ("mask_count","n_points","n"):
        v = getattr(r, attr, None)
        if v is not None: return int(v)
    return int(fallback_len)

def _get_window_k(r):
    wk = getattr(r, "window_k", None)
    if isinstance(wk, (list, tuple)) and len(wk) == 2:
        try: return [float(wk[0]), float(wk[1])]
        except Exception: return None
    return None

def main():
    ap = argparse.ArgumentParser(description="Per-IFO slope2 fit per event")
    ap.add_argument("--data", default="data/ct/ct_bounds.csv")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--event", default=None)
    ap.add_argument("--method", default="wls", choices=["wls","huber"])
    args = ap.parse_args()

    df = load_ct(Path(args.data))
    if args.event:
        df = df[df["event"] == args.event]

    by_event = {}
    for ev in sorted(df["event"].unique()):
        dfe = df[df["event"] == ev].copy()
        if dfe.empty: continue

        tmp_all = Path("data/ct/_tmp_all.csv")
        dfe.to_csv(tmp_all, index=False)
        r_all = do_fit(tmp_all, Path(args.profile) if args.profile else None, method=args.method)

        perifo = {}
        for ifo in sorted(dfe["ifo"].unique()):
            dfi = dfe[dfe["ifo"] == ifo].copy()
            if dfi.empty: continue
            tmp_1 = Path("data/ct/_tmp_1.csv")
            dfi.to_csv(tmp_1, index=False)
            r = do_fit(tmp_1, Path(args.profile) if args.profile else None, method=args.method)
            perifo[ifo] = {
                "slope": float(r.slope),
                "intercept_log10": _get_intercept_any(r),
                "n": _get_n_used(r, len(dfi)),
                "method": args.method,
                "window_k": _get_window_k(r)
            }

        by_event[ev] = {
            "per_ifo": perifo,
            "combined": {
                "slope": float(r_all.slope),
                "intercept_log10": _get_intercept_any(r_all),
                "n": _get_n_used(r_all, len(dfe)),
                "method": args.method,
                "window_k": _get_window_k(r_all)
            }
        }

    out = {"by_event": by_event, "overall": {"n_events": len(by_event)}}
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/slope2_perifo.json").write_text(json.dumps(out, indent=2))
    print("Wrote reports/slope2_perifo.json")

if __name__ == "__main__":
    main()
