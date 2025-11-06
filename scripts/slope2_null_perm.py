# scripts/slope2_null_perm.py
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from uclgw.eval.slopefit import load_ct, do_fit, _read_profile_window

EPS_FLOOR = 1e-14

def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "delta_ct2" in df.columns:
        df["delta_ct2"] = np.abs(df["delta_ct2"]).clip(lower=EPS_FLOOR)
    return df

def main():
    ap = argparse.ArgumentParser(description="Permutation null for slope-2")
    ap.add_argument("--data", default="data/ct/ct_bounds.csv")
    ap.add_argument("--profile", default="configs/profiles/lvk_o3.yaml")
    ap.add_argument("--event", default=None)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--method", default="wls", choices=["wls","huber"])
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    kmin, kmax = _read_profile_window(Path(args.profile))

    df = load_ct(Path(args.data))
    if args.event:
        df = df[df["event"] == args.event]
    df = _sanitize_df(df)
    m = np.isfinite(df["k"].values) & np.isfinite(df["delta_ct2"].values)
    m &= (df["k"].values >= kmin) & (df["k"].values <= kmax)
    df = df[m].reset_index(drop=True)

    if len(df) < 8:
        print("Too few points after filtering; aborting.")
        return

    # 真實 slope
    tmp = Path("data/ct/_tmp_real.csv")
    df.to_csv(tmp, index=False)
    r_real = do_fit(tmp, Path(args.profile), method=args.method)
    s_real = float(r_real.slope)

    # 置換：打亂 k 與 delta_ct2 的對應
    s_perm = []
    for _ in range(args.n_perm):
        dfp = df.copy()
        dfp["k"] = rng.permutation(dfp["k"].values)
        tmp = Path("data/ct/_tmp_perm.csv")
        dfp.to_csv(tmp, index=False)
        rp = do_fit(tmp, Path(args.profile), method=args.method)
        s_perm.append(float(rp.slope))
    s_perm = np.array(s_perm, dtype=float)

    p = float((s_perm >= s_real).mean())  # 一尾（期望 +2）

    out = {
        "event": args.event,
        "method": args.method,
        "window_k": [float(kmin), float(kmax)],
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
