# scripts/slope2_perifo.py
# --- add project src to path ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
# --------------------------------

import argparse, json
import pandas as pd
from uclgw.eval.slopefit import load_ct, do_fit

def _result_n_used(result, fallback_len: int) -> int:
    n_used = getattr(result, "mask_count", None)
    if n_used is None:
        n_used = getattr(result, "n_points", None)
    if n_used is None:
        n_used = fallback_len
    return int(n_used)

def _result_window_k(result):
    wk = getattr(result, "window_k", None)
    if isinstance(wk, (list, tuple)):
        try:
            return list(map(float, wk))
        except Exception:
            return None
    return None

def main():
    ap = argparse.ArgumentParser(description="Per-IFO slope2 fit per event (WLS).")
    ap.add_argument("--data", default="data/ct/ct_bounds.csv")
    ap.add_argument("--profile", default="configs/profiles/lvk_o3.yaml")
    args = ap.parse_args()

    df = load_ct(Path(args.data))
    events = sorted(df["event"].unique())

    by_event = {}
    for ev in events:
        dfe = df[df["event"] == ev].copy()
        if dfe.empty: continue

        # 合併（全 IFO）
        tmp_all = Path("data/ct/_tmp_real.csv")
        dfe.to_csv(tmp_all, index=False)
        r_all = do_fit(tmp_all, Path(args.profile), method="wls")

        perifo = {}
        for ifo in sorted(dfe["ifo"].unique()):
            dfi = dfe[dfe["ifo"] == ifo].copy()
            if dfi.empty: continue
            tmp_1 = Path("data/ct/_tmp_real.csv")
            dfi.to_csv(tmp_1, index=False)
            r = do_fit(tmp_1, Path(args.profile), method="wls")
            n_used = _result_n_used(r, fallback_len=len(dfi))
            perifo[ifo] = {
                "slope": float(r.slope),
                "intercept": float(r.intercept),
                "n": n_used,
                "method": "wls",
                "mask_count": n_used,
                "window_k": _result_window_k(r)
            }

        by_event[ev] = {
            "per_ifo": perifo,
            "combined_wls": {
                "slope": float(r_all.slope),
                "intercept": float(r_all.intercept),
                "n": _result_n_used(r_all, fallback_len=len(dfe)),
                "method": "wls",
                "mask_count": _result_n_used(r_all, fallback_len=len(dfe)),
                "window_k": _result_window_k(r_all)
            }
        }

    out = {"by_event": by_event, "overall": {"n_events": len(by_event)}}
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/slope2_perifo.json").write_text(json.dumps(out, indent=2))
    print("Wrote reports/slope2_perifo.json")

if __name__ == "__main__":
    main()
