# scripts/slope2_null_perm.py
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from uclgw.eval.slopefit import load_ct, do_fit, _read_profile_window


def _finite_mask(df):
    m = np.isfinite(df["k"].values) & np.isfinite(df["delta_ct2"].values)
    return m


def _window_mask(df, kmin, kmax):
    return (df["k"] >= kmin) & (df["k"] <= kmax) & _finite_mask(df)


def main():
    ap = argparse.ArgumentParser(description="Permutation null for slope-2 (profile-windowed).")
    ap.add_argument("--data", default="data/ct/ct_bounds.csv")
    ap.add_argument("--profile", default="configs/profiles/lvk_o3.yaml")
    ap.add_argument("--event", default=None)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--method", default="wls", choices=["wls", "huber"])
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = load_ct(Path(args.data))
    if args.event:
        df = df[df["event"] == args.event]

    kmin, kmax = _read_profile_window(Path(args.profile))
    m = _window_mask(df, kmin, kmax)
    df = df[m].reset_index(drop=True)

    if len(df) < 8:
        raise SystemExit("Not enough points after windowing for permutation test.")

    Path("reports").mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("data/ct")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 真實 slope（已視窗裁切；為謹慎仍傳 profile）
    tmp_real = tmp_dir / "_tmp_perm_real.csv"
    df.to_csv(tmp_real, index=False)
    r_real = do_fit(tmp_real, Path(args.profile), method=args.method)
    s_real = float(getattr(r_real, "slope"))

    # 置換：打亂 k 與 delta_ct2 的關聯
    s_perm = np.empty(args.n_perm, dtype=float)
    tmp_perm = tmp_dir / "_tmp_perm_perm.csv"
    for i in range(args.n_perm):
        dfp = df.copy()
        dfp["k"] = rng.permutation(dfp["k"].values)
        dfp.to_csv(tmp_perm, index=False)
        rp = do_fit(tmp_perm, Path(args.profile), method=args.method)
        s_perm[i] = float(getattr(rp, "slope"))

    # 單尾 p-value：null ≥ s_real 的比例（測試「是否顯著偏大」）
    p_one = float(np.mean(s_perm >= s_real))

    out = {
        "event": args.event,
        "method": args.method,
        "window_k": [float(kmin), float(kmax)],
        "slope_real": float(s_real),
        "perm_mean": float(np.nanmean(s_perm)),
        "perm_std": float(np.nanstd(s_perm, ddof=1)),
        "p_value_one_sided": p_one,
        "n_perm": int(args.n_perm),
    }
    Path("reports/slope2_null_perm.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("Wrote reports/slope2_null_perm.json")


if __name__ == "__main__":
    main()
