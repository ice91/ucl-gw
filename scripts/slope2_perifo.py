# scripts/slope2_perifo.py
# --- add project src to path ---
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
# --------------------------------

import numpy as np
import pandas as pd
import yaml
from uclgw.eval.slopefit import load_ct, do_fit

C = 299_792_458.0
def _finite_mask(df: pd.DataFrame) -> np.ndarray:
    return np.isfinite(df["k"].values) & np.isfinite(df["delta_ct2"].values)

def _read_profile_fwindow(profile_path: Path | None):
    if profile_path is None:
        return None, None
    prof = yaml.safe_load(Path(profile_path).read_text())
    fmin = prof.get("fmin", None)
    fmax = prof.get("fmax", None)
    return (float(fmin) if fmin is not None else None,
            float(fmax) if fmax is not None else None)

def _k_from_f(f: float) -> float:
    return 2*np.pi*float(f)/C

def _n_used(result, fallback_len: int) -> int:
    try:
        return int(len(result.x))
    except Exception:
        return int(fallback_len)

def main():
    ap = argparse.ArgumentParser(description="Per-IFO slope2 fit per event (WLS/Huber, profile-windowed).")
    ap.add_argument("--data", default="data/ct/ct_bounds.csv")
    ap.add_argument("--profile", default="configs/profiles/lvk_o3.yaml")
    ap.add_argument("--event", default=None)
    ap.add_argument("--method", default="wls", choices=["wls", "huber"])
    args = ap.parse_args()

    df = load_ct(Path(args.data))
    if args.event:
        df = df[df["event"] == args.event]
    df = df[_finite_mask(df)].copy()
    if df.empty:
        raise SystemExit("No valid rows after finite filtering.")

    fmin, fmax = _read_profile_fwindow(Path(args.profile))
    kmin = _k_from_f(fmin) if fmin is not None else float(df["k"].min())
    kmax = _k_from_f(fmax) if fmax is not None else float(df["k"].max())

    events = sorted(df["event"].unique())
    by_event = {}
    tmp_dir = Path("data/ct"); tmp_dir.mkdir(parents=True, exist_ok=True)

    for ev in events:
        dfe = df[df["event"] == ev].copy()
        tmp_all = tmp_dir / f"_tmp_perifo_{ev}_ALL.csv"
        dfe.to_csv(tmp_all, index=False)
        # do_fit 內部會依 profile 的 fmin/fmax 以 f_hz 欄位遮罩
        r_all = do_fit(tmp_all, str(args.profile), method=args.method)

        perifo = {}
        for ifo in sorted(dfe["ifo"].unique()):
            dfi = dfe[dfe["ifo"] == ifo].copy()
            if dfi.empty:
                continue
            tmp_1 = tmp_dir / f"_tmp_perifo_{ev}_{ifo}.csv"
            dfi.to_csv(tmp_1, index=False)
            r = do_fit(tmp_1, str(args.profile), method=args.method)
            perifo[ifo] = {
                "slope": float(r.slope),
                "intercept_log10": float(r.intercept),  # 截距已在 log10 空間
                "n": _n_used(r, fallback_len=len(dfi)),
                "method": args.method,
                "window_k": [float(kmin), float(kmax)],
            }

        by_event[ev] = {
            "per_ifo": perifo,
            "combined": {
                "slope": float(r_all.slope),
                "intercept_log10": float(r_all.intercept),
                "n": _n_used(r_all, fallback_len=len(dfe)),
                "method": args.method,
                "window_k": [float(kmin), float(kmax)],
            },
        }

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/slope2_perifo.json").write_text(json.dumps(
        {"by_event": by_event, "overall": {"n_events": len(by_event)}},
        indent=2, ensure_ascii=False
    ))
    print("Wrote reports/slope2_perifo.json")

if __name__ == "__main__":
    main()
