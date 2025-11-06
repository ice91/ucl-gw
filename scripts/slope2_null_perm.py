# scripts/slope2_null_perm.py
import sys, argparse, json
from pathlib import Path
import numpy as np
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
            # 回退：若只有 fmin/fmax，就轉 k
            if "fmin" in cfg and "fmax" in cfg:
                c = 299792458.0
                kmin = 2*np.pi*float(cfg["fmin"])/c
                kmax = 2*np.pi*float(cfg["fmax"])/c
                return [float(kmin), float(kmax)]
    except Exception:
        pass
    return None

def _finite_mask(df):
    import numpy as np
    cols = ["k","delta_ct2"]
    m = np.isfinite(df[cols]).all(axis=1)
    return m

def main():
    ap = argparse.ArgumentParser(description="Permutation null for slope-2")
    ap.add_argument("--data", default="data/ct/ct_bounds.csv")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--event", default=None)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--method", default="wls", choices=["wls","huber"])
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = load_ct(Path(args.data))
    if args.event:
        df = df[df["event"] == args.event].copy()

    wk = _read_window_from_profile(args.profile)
    if wk:
        df = df[(df["k"] >= wk[0]) & (df["k"] <= wk[1])].copy()
    df = df[_finite_mask(df)].reset_index(drop=True)

    if len(df) < 8:
        raise SystemExit("Not enough points after filtering.")

    # 真實 slope
    tmp = Path("data/ct/_tmp_real.csv"); tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    r_real = do_fit(tmp, Path(args.profile) if args.profile else None, method=args.method)
    s_real = float(r_real.slope)

    # 置換：打亂 k 與 delta_ct2 的對應
    s_perm = []
    for _ in range(args.n_perm):
        dfp = df.copy()
        dfp["k"] = rng.permutation(dfp["k"].values)
        tperm = Path("data/ct/_tmp_perm.csv")
        dfp.to_csv(tperm, index=False)
        rp = do_fit(tperm, Path(args.profile) if args.profile else None, method=args.method)
        s_perm.append(float(rp.slope))
    s_perm = np.asarray(s_perm)

    p = float((s_perm >= s_real).mean())

    out = {
        "event": args.event,
        "method": args.method,
        "window_k": wk,
        "slope_real": s_real,
        "perm_mean": float(s_perm.mean()),
        "perm_std": float(s_perm.std(ddof=1)),
        "p_value_one_sided": p,
        "n_perm": int(args.n_perm)
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/slope2_null_perm.json").write_text(json.dumps(out, indent=2))
    print("Wrote reports/slope2_null_perm.json")

if __name__ == "__main__":
    main()
