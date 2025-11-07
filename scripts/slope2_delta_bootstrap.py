# scripts/slope2_delta_bootstrap.py
import sys, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from uclgw.eval.slopefit import load_ct, do_fit

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

def _apply_window(df: pd.DataFrame, wk):
    if wk and len(wk)==2:
        df = df[(df["k"] >= wk[0]) & (df["k"] <= wk[1])].copy()
    return df

def _fit_slope(df: pd.DataFrame, profile, method: str) -> float:
    tmp = Path("data/ct/_tmp_bs.csv"); tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    r = do_fit(tmp, Path(profile) if profile else None, method=method)
    return float(r.slope)

def main():
    ap = argparse.ArgumentParser(description="Bootstrap test for Δslope = slope(REAL) - slope(OFF)")
    ap.add_argument("--data-real", required=True)
    ap.add_argument("--event-real", required=True)
    ap.add_argument("--data-off", required=True)
    ap.add_argument("--event-off", required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--method", default="huber", choices=["wls","huber"])
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--preclean", action="store_true")
    ap.add_argument("--sigma-quantiles", default="0.01,0.99")
    ap.add_argument("--zmax", type=float, default=3.5)
    ap.add_argument("--ct2-q-hi", type=float, default=0.99)
    ap.add_argument("--uniform-weight", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    dfr = load_ct(Path(args.data_real))
    dfr = dfr[dfr["event"] == args.event_real].copy()
    dfo = load_ct(Path(args.data_off))
    dfo = dfo[dfo["event"] == args.event_off].copy()

    if dfr.empty or dfo.empty:
        print(f"[ERROR] Empty REAL or OFF after event filter. REAL rows={len(dfr)}, OFF rows={len(dfo)}")
        raise SystemExit(2)

    wk = _read_window_from_profile(args.profile)
    dfr = _apply_window(dfr, wk)
    dfo = _apply_window(dfo, wk)

    if args.preclean:
        lo, hi = map(float, args.sigma_quantiles.split(","))
        dfr = _preclean(dfr, lo, hi, args.zmax, args.ct2_q_hi)
        dfo = _preclean(dfo, lo, hi, args.zmax, args.ct2_q_hi)

    if args.uniform_weight:
        if "sigma" in dfr: dfr = dfr.copy(); dfr["sigma"] = 1.0
        if "sigma" in dfo: dfo = dfo.copy(); dfo["sigma"] = 1.0

    if len(dfr) < 8 or len(dfo) < 8:
        raise SystemExit("Not enough points after filtering for bootstrap.")

    s_real = _fit_slope(dfr, args.profile, args.method)
    s_off  = _fit_slope(dfo, args.profile, args.method)
    delta  = s_real - s_off

    # bootstrap：各自 with replacement 取與原表同長度
    deltas = []
    for _ in range(args.n_bootstrap):
        br = dfr.sample(n=len(dfr), replace=True, random_state=rng.integers(0, 2**31-1))
        bo = dfo.sample(n=len(dfo), replace=True, random_state=rng.integers(0, 2**31-1))
        sr = _fit_slope(br, args.profile, args.method)
        so = _fit_slope(bo, args.profile, args.method)
        deltas.append(sr - so)
    deltas = np.asarray(deltas)

    # 兩側 p：看 |Δ*| >= |Δ|
    p_two = float((np.abs(deltas) >= abs(delta)).mean())
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])

    out = {
        "event_real": args.event_real,
        "event_off": args.event_off,
        "method": args.method,
        "window_k": wk,
        "slope_real": float(s_real),
        "slope_off": float(s_off),
        "delta_slope": float(delta),
        "bootstrap_ci95": [float(ci_lo), float(ci_hi)],
        "p_value_two_sided": p_two,
        "n_bootstrap": int(args.n_bootstrap),
        "preclean": bool(args.preclean),
        "uniform_weight": bool(args.uniform_weight)
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/slope2_delta_bootstrap.json").write_text(json.dumps(out, indent=2))
    print("Wrote reports/slope2_delta_bootstrap.json")

if __name__ == "__main__":
    main()
