# scripts/slope2_perifo.py
import sys, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

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

def _finite_mask(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["k","delta_ct2","sigma"] if c in df.columns]
    m = np.isfinite(df[cols]).all(axis=1)
    return m

def _preclean(df: pd.DataFrame, sigma_q_lo: float, sigma_q_hi: float,
              zmax: float, ct2_q_hi: float) -> pd.DataFrame:
    df = df[_finite_mask(df)].copy()
    if df.empty: return df
    # sigma 量化截尾
    lo = df["sigma"].quantile(sigma_q_lo) if "sigma" in df else None
    hi = df["sigma"].quantile(sigma_q_hi) if "sigma" in df else None
    ms = np.ones(len(df), dtype=bool)
    if lo is not None and hi is not None:
        ms = (df["sigma"] >= lo) & (df["sigma"] <= hi)
    # log10(delta_ct2) z-score
    logy = np.log10(df["delta_ct2"].clip(lower=1e-18))
    mu, sd = np.mean(logy), np.std(logy, ddof=1)
    mz = np.ones(len(df), dtype=bool) if not np.isfinite(sd) or sd == 0 \
        else (np.abs((logy - mu) / sd) <= zmax)
    # 上尾去極端（原尺度）
    qhi = df["delta_ct2"].quantile(ct2_q_hi)
    mt = df["delta_ct2"] <= max(float(qhi), 1e-18)
    m = ms & mz & mt
    return df[m].copy()

def _apply_window(df: pd.DataFrame, window_k):
    if window_k and len(window_k) == 2:
        kmin, kmax = float(window_k[0]), float(window_k[1])
        df = df[(df["k"] >= kmin) & (df["k"] <= kmax)].copy()
    return df

def _read_window_from_profile(profile_path):
    if not profile_path: return None
    try:
        import yaml, numpy as _np
        cfg = yaml.safe_load(Path(profile_path).read_text())
        if isinstance(cfg, dict):
            if "window_k" in cfg and isinstance(cfg["window_k"], (list, tuple)) and len(cfg["window_k"])==2:
                return [float(cfg["window_k"][0]), float(cfg["window_k"][1])]
            if "fmin" in cfg and "fmax" in cfg:
                c = 299792458.0
                kmin = 2*_np.pi*float(cfg["fmin"])/c
                kmax = 2*_np.pi*float(cfg["fmax"])/c
                return [float(kmin), float(kmax)]
    except Exception:
        pass
    return None

def main():
    ap = argparse.ArgumentParser(description="Per-IFO slope2 fit per event (with optional preclean & uniform-weight)")
    ap.add_argument("--data", default="data/ct/ct_bounds.csv")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--event", default=None)
    ap.add_argument("--method", default="wls", choices=["wls","huber"])
    ap.add_argument("--preclean", action="store_true", help="Enable minimal robust cleaning")
    ap.add_argument("--sigma-quantiles", default="0.01,0.99", help="lo,hi for sigma trimming")
    ap.add_argument("--zmax", type=float, default=3.5, help="Z-score threshold on log10(delta_ct2)")
    ap.add_argument("--ct2-q-hi", type=float, default=0.99, help="Upper-tail quantile for delta_ct2")
    ap.add_argument("--uniform-weight", action="store_true", help="Override sigma to 1.0 before fitting")
    args = ap.parse_args()

    df = load_ct(Path(args.data))
    if args.event:
        df = df[df["event"] == args.event].copy()
        if df.empty:
            # 列出可用事件，避免空結果誤解
            all_events = sorted(load_ct(Path(args.data))["event"].unique().tolist())
            print(f"[ERROR] Event '{args.event}' not found in {args.data}. Available: {all_events}")
            raise SystemExit(2)

    # 統一視窗（若提供 profile）
    wk = _read_window_from_profile(args.profile)

    by_event = {}
    for ev in sorted(df["event"].unique()):
        dfe = df[df["event"] == ev].copy()
        if dfe.empty: continue
        dfe = _apply_window(dfe, wk)
        if args.preclean:
            dfe = _preclean(dfe, *map(float, args.sigma_quantiles.split(",")), args.zmax, args.ct2_q_hi)
        if args.uniform_weight and "sigma" in dfe:
            dfe = dfe.copy(); dfe["sigma"] = 1.0

        if dfe.empty: 
            continue

        tmp_all = Path("data/ct/_tmp_all.csv"); tmp_all.parent.mkdir(parents=True, exist_ok=True)
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
                "window_k": _get_window_k(r) or wk
            }

        by_event[ev] = {
            "per_ifo": perifo,
            "combined": {
                "slope": float(r_all.slope),
                "intercept_log10": _get_intercept_any(r_all),
                "n": _get_n_used(r_all, len(dfe)),
                "method": args.method,
                "window_k": _get_window_k(r_all) or wk
            }
        }

    out = {"by_event": by_event, "overall": {"n_events": len(by_event)}}
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/slope2_perifo.json").write_text(json.dumps(out, indent=2))
    print("Wrote reports/slope2_perifo.json")

if __name__ == "__main__":
    main()
