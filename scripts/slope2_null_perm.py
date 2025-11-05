# scripts/slope2_null_perm.py
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from uclgw.eval.slopefit import load_ct, do_fit, window_mask, _read_profile_window

def main():
    ap = argparse.ArgumentParser(description="Permutation null for slope-2")
    ap.add_argument("--data", default="data/ct/ct_bounds.csv")
    ap.add_argument("--profile", default="configs/profiles/lvk_o3.yaml")
    ap.add_argument("--event", default=None)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = load_ct(Path(args.data))
    if args.event:
        df = df[df["event"] == args.event]
    kmin, kmax = _read_profile_window(Path(args.profile))
    m = window_mask(df, kmin, kmax)
    df = df[m].reset_index(drop=True)

    # 真實 slope
    tmp = Path("data/ct/_tmp_real.csv")
    df.to_csv(tmp, index=False)
    r_real = do_fit(tmp, None, method="wls")
    s_real = r_real.slope

    # 置換：打亂 (k,delta_ct2) 的對應，破壞物理關聯
    s_perm = []
    for _ in range(args.n_perm):
        dfp = df.copy()
        dfp["k"] = rng.permutation(dfp["k"].values)
        tmp = Path("data/ct/_tmp_perm.csv")
        dfp.to_csv(tmp, index=False)
        rp = do_fit(tmp, None, method="wls")
        s_perm.append(rp.slope)
    s_perm = np.array(s_perm)
    # p-value（單尾）：看有多少 null ≥ s_real
    p = float((s_perm >= s_real).mean())

    out = {
        "slope_real": float(s_real),
        "perm_mean": float(s_perm.mean()),
        "perm_std": float(s_perm.std(ddof=1)),
        "p_value_one_sided": p,
        "n_perm": args.n_perm
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/slope2_null_perm.json").write_text(json.dumps(out, indent=2))
    print("Wrote reports/slope2_null_perm.json")

if __name__ == "__main__":
    main()
