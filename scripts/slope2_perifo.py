# scripts/slope2_perifo.py
# --- add project src to path ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
# --------------------------------

import argparse, json
import numpy as np
import pandas as pd
from uclgw.eval.slopefit import load_ct, do_fit, _read_profile_window


def _finite_mask(df: pd.DataFrame) -> np.ndarray:
    cols = ["k", "delta_ct2"]
    m = np.ones(len(df), dtype=bool)
    for c in cols:
        m &= np.isfinite(df[c].values)
    return m


def _result_n_used(result, fallback_len: int) -> int:
    for attr in ("n_points", "mask_count"):
        v = getattr(result, attr, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return int(fallback_len)


def main():
    ap = argparse.ArgumentParser(description="Per-IFO slope2 fit per event (same window/weight as profile).")
    ap.add_argument("--data", default="data/ct/ct_bounds.csv")
    ap.add_argument("--profile", default="configs/profiles/lvk_o3.yaml")
    ap.add_argument("--event", default=None, help="Restrict to a single event (optional).")
    ap.add_argument("--method", default="wls", choices=["wls", "huber"], help="Regression method.")
    args = ap.parse_args()

    df = load_ct(Path(args.data))
    if args.event:
        df = df[df["event"] == args.event]

    # 全域有限值過濾（保守）
    df = df[_finite_mask(df)].copy()
    if df.empty:
        raise SystemExit("No valid rows after finite filtering.")

    # 讀取 profile 視窗並套用到所有回歸
    kmin, kmax = _read_profile_window(Path(args.profile))
    def _window(df_):
        return df_[(df_["k"] >= kmin) & (df_["k"] <= kmax) & _finite_mask(df_)].copy()

    events = sorted(df["event"].unique())
    by_event = {}

    # 暫存檔資料夾
    tmp_dir = Path("data/ct")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for ev in events:
        dfe = _window(df[df["event"] == ev])
        if dfe.empty:
            continue

        # 合併（全 IFO）
        tmp_all = tmp_dir / f"_tmp_perifo_{ev}_ALL.csv"
        dfe.to_csv(tmp_all, index=False)
        r_all = do_fit(tmp_all, Path(args.profile), method=args.method)

        perifo = {}
        for ifo in sorted(dfe["ifo"].unique()):
            dfi = _window(dfe[dfe["ifo"] == ifo])
            if dfi.empty:
                continue
            tmp_1 = tmp_dir / f"_tmp_perifo_{ev}_{ifo}.csv"
            dfi.to_csv(tmp_1, index=False)
            r = do_fit(tmp_1, Path(args.profile), method=args.method)
            perifo[ifo] = {
                "slope": float(getattr(r, "slope")),
                "intercept_log10": float(getattr(r, "intercept_log10")),
                "n": _result_n_used(r, fallback_len=len(dfi)),
                "method": args.method,
                "window_k": [float(kmin), float(kmax)],
            }

        by_event[ev] = {
            "per_ifo": perifo,
            "combined": {
                "slope": float(getattr(r_all, "slope")),
                "intercept_log10": float(getattr(r_all, "intercept_log10")),
                "n": _result_n_used(r_all, fallback_len=len(dfe)),
                "method": args.method,
                "window_k": [float(kmin), float(kmax)],
            },
        }

    out = {"by_event": by_event, "overall": {"n_events": len(by_event)}}
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/slope2_perifo.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("Wrote reports/slope2_perifo.json")


if __name__ == "__main__":
    main()
