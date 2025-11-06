# scripts/slope2_jackknife.py
# --- add project src to path ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
# --------------------------------

import argparse, json
import numpy as np
import pandas as pd
import yaml
from uclgw.eval.slopefit import load_ct, do_fit

def _read_profile_window(profile_path: Path) -> tuple[float, float] | None:
    p = Path(profile_path)
    if not p.exists(): return None
    y = yaml.safe_load(p.read_text())
    win = y.get("window_k", None)
    if isinstance(win, (list, tuple)) and len(win) == 2:
        try:
            kmin = float(win[0]); kmax = float(win[1]); return (kmin, kmax)
        except Exception:
            return None
    return None

def _apply_window(df: pd.DataFrame, k_window: tuple[float,float] | None) -> pd.DataFrame:
    if k_window is None: return df
    kmin, kmax = k_window
    return df[(df["k"] >= kmin) & (df["k"] <= kmax)].copy()

def main():
    ap = argparse.ArgumentParser(description="Jackknife/Bootstrap uncertainty for slope2 (single event).")
    ap.add_argument("--data", default="data/ct/ct_bounds.csv")
    ap.add_argument("--profile", default="configs/profiles/lvk_o3.yaml")
    ap.add_argument("--event", required=True, help="Target event (required for single-event mode)")
    ap.add_argument("--n-bootstrap", type=int, default=400)
    args = ap.parse_args()

    df = load_ct(Path(args.data))
    df = df[df["event"] == args.event].copy()
    if df.empty:
        raise SystemExit(f"No rows for event={args.event}")

    k_window = _read_profile_window(Path(args.profile))
    df = _apply_window(df, k_window)
    if df.empty:
        raise SystemExit("All rows masked by window_k")

    # 基準 fit
    tmp = Path("data/ct/_tmp_jk.csv")
    df.to_csv(tmp, index=False)
    r0 = do_fit(tmp, Path(args.profile), method="wls")

    # Jackknife: leave-one-out
    slopes = []
    idx = np.arange(df.shape[0])
    for i in range(df.shape[0]):
        mask = idx != i
        dfi = df.iloc[mask].copy()
        if dfi.shape[0] < 3: continue
        dfi.to_csv(tmp, index=False)
        ri = do_fit(tmp, Path(args.profile), method="wls")
        slopes.append(float(ri.slope))

    if len(slopes) == 0:
        raise SystemExit("Jackknife failed (not enough points).")

    slopes = np.array(slopes)
    jk_mean = float(slopes.mean())
    jk_se = float(np.sqrt((len(slopes) - 1) / len(slopes) * np.sum((slopes - jk_mean)**2)))

    # Bootstrap
    boot = []
    n = df.shape[0]
    for _ in range(int(args.n_bootstrap)):
        take = np.random.randint(0, n, size=n)
        dfi = df.iloc[take].copy()
        dfi.to_csv(tmp, index=False)
        ri = do_fit(tmp, Path(args.profile), method="wls")
        boot.append(float(ri.slope))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    out = {
        "n_points": int(df.shape[0]),
        "jk_mean": jk_mean,
        "jk_se": jk_se,
        "bootstrap_ci95": [float(lo), float(hi)],
        "bootstrap_n": int(args.n_bootstrap),
        "ref_slope": float(r0.slope),
        "window_k": list(map(float, k_window)) if k_window else None
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/slope2_jackknife.json").write_text(json.dumps(out, indent=2))
    print("Wrote reports/slope2_jackknife.json")

if __name__ == "__main__":
    main()
