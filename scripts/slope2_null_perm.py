# scripts/slope2_null_perm.py
import argparse, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

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
    df = df[_finite_mask(df)].copy()

    # 依 profile 視窗裁切
    fmin, fmax = _read_profile_fwindow(Path(args.profile))
    if fmin is not None: df = df[df["f_hz"] >= fmin]
    if fmax is not None: df = df[df["f_hz"] <= fmax]
    df = df.reset_index(drop=True)

    if len(df) < 8:
        raise SystemExit("Not enough points after windowing for permutation test.")

    Path("reports").mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("data/ct"); tmp_dir.mkdir(parents=True, exist_ok=True)

    # 真實 slope
    tmp_real = tmp_dir / "_tmp_perm_real.csv"
    df.to_csv(tmp_real, index=False)
    r_real = do_fit(tmp_real, None, method=args.method)  # 已裁切，傳 None
    s_real = float(r_real.slope)

    # 置換：打亂 k 與 delta_ct2 的對應（保留邊際分佈）
    s_perm = np.empty(args.n_perm, dtype=float)
    tmp_perm = tmp_dir / "_tmp_perm_perm.csv"
    for i in range(args.n_perm):
        dfp = df.copy()
        dfp["k"] = rng.permutation(dfp["k"].values)
        dfp.to_csv(tmp_perm, index=False)
        rp = do_fit(tmp_perm, None, method=args.method)
        s_perm[i] = float(rp.slope)

    p_one = float(np.mean(s_perm >= s_real))  # 偏大的一尾檢定（期望 +2）

    out = {
        "event": args.event,
        "method": args.method,
        "window_k": [
            float(_k_from_f(fmin)) if fmin is not None else float(df["k"].min()),
            float(_k_from_f(fmax)) if fmax is not None else float(df["k"].max())
        ],
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
