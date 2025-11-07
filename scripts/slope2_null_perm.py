# scripts/slope2_null_perm.py
import sys, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from uclgw.eval.slopefit import load_ct, do_fit

def _read_window_from_profile(profile_path):
    if not profile_path: return None
    try:
        import yaml
        cfg = yaml.safe_load(Path(profile_path).read_text())
        if isinstance(cfg, dict):
            if "window_k" in cfg and isinstance(cfg["window_k"], (list, tuple)) and len(cfg["window_k"])==2:
                return [float(cfg["window_k"][0]), float(cfg["window_k"][1])]
            if "fmin" in cfg and "fmax" in cfg:
                c = 299792458.0
                kmin = 2*np.pi*float(cfg["fmin"])/c
                kmax = 2*np.pi*float(cfg["fmax"])/c
                return [float(kmin), float(kmax)]
    except Exception:
        pass
    return None

def _finite_mask(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["k","delta_ct2","sigma"] if c in df.columns]
    return np.isfinite(df[cols]).all(axis=1)

def _preclean(df: pd.DataFrame, sigma_q_lo: float, sigma_q_hi: float,
              zmax: float, ct2_q_hi: float) -> pd.DataFrame:
    df = df[_finite_mask(df)].copy()
    if df.empty: return df
    lo = df["sigma"].quantile(sigma_q_lo) if "sigma" in df else None
    hi = df["sigma"].quantile(sigma_q_hi) if "sigma" in df else None
    ms = np.ones(len(df), dtype=bool)
    if lo is not None and hi is not None:
        ms = (df["sigma"] >= lo) & (df["sigma"] <= hi)
    logy = np.log10(df["delta_ct2"].clip(lower=1e-18))
    mu, sd = np.mean(logy), np.std(logy, ddof=1)
    mz = np.ones(len(df), dtype=bool) if not np.isfinite(sd) or sd == 0 \
        else (np.abs((logy - mu) / sd) <= zmax)
    qhi = df["delta_ct2"].quantile(ct2_q_hi)
    mt = df["delta_ct2"] <= max(float(qhi), 1e-18)
    return df[ms & mz & mt].copy()

def main():
    ap = argparse.ArgumentParser(description="Permutation null for slope-2 (with preclean & uniform-weight)")
    ap.add_argument("--data", default="data/ct/ct_bounds.csv")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--event", default=None)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--method", default="wls", choices=["wls","huber"])
    ap.add_argument("--two-sided", action="store_true", help="Report two-sided p as well")
    ap.add_argument("--preclean", action="store_true")
    ap.add_argument("--sigma-quantiles", default="0.01,0.99")
    ap.add_argument("--zmax", type=float, default=3.5)
    ap.add_argument("--ct2-q-hi", type=float, default=0.99)
    ap.add_argument("--uniform-weight", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = load_ct(Path(args.data))
    if args.event:
        df = df[df["event"] == args.event].copy()
        if df.empty:
            all_events = sorted(load_ct(Path(args.data))["event"].unique().tolist())
            print(f"[ERROR] Event '{args.event}' not found in {args.data}. Available: {all_events}")
            raise SystemExit(2)

    wk = _read_window_from_profile(args.profile)
    if wk:
        df = df[(df["k"] >= wk[0]) & (df["k"] <= wk[1])].copy()
    df = df[_finite_mask(df)].reset_index(drop=True)

    if args.preclean:
        lo, hi = map(float, args.sigma_quantiles.split(","))
        df = _preclean(df, lo, hi, args.zmax, args.ct2_q_hi)

    if args.uniform_weight and "sigma" in df:
        df = df.copy(); df["sigma"] = 1.0

    if len(df) < 8:
        raise SystemExit("Not enough points after filtering.")

    tmp = Path("data/ct/_tmp_real.csv"); tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    r_real = do_fit(tmp, Path(args.profile) if args.profile else None, method=args.method)
    s_real = float(r_real.slope)

    s_perm = []
    for _ in range(args.n_perm):
        dfp = df.copy()
        dfp["k"] = rng.permutation(dfp["k"].values)
        tperm = Path("data/ct/_tmp_perm.csv")
        dfp.to_csv(tperm, index=False)
        rp = do_fit(tperm, Path(args.profile) if args.profile else None, method=args.method)
        s_perm.append(float(rp.slope))
    s_perm = np.asarray(s_perm)

    # 單尾（依 s_real 符號）；雙尾
    if s_real >= 0:
        p_one = float((s_perm >= s_real).mean())
    else:
        p_one = float((s_perm <= s_real).mean())
    p_two = float((np.abs(s_perm) >= abs(s_real)).mean()) if args.two_sided else None

    out = {
        "event": args.event,
        "method": args.method,
        "window_k": wk,
        "slope_real": s_real,
        "perm_mean": float(s_perm.mean()),
        "perm_std": float(s_perm.std(ddof=1)),
        "p_value_one_sided": p_one,
        "p_value_two_sided": p_two,
        "n_perm": int(args.n_perm),
        "preclean": bool(args.preclean),
        "uniform_weight": bool(args.uniform_weight)
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/slope2_null_perm.json").write_text(json.dumps(out, indent=2))
    print("Wrote reports/slope2_null_perm.json")

if __name__ == "__main__":
    main()
