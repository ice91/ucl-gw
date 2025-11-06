# scripts/slope2_robustness.py
import argparse, json, sys, itertools
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from uclgw.eval.slopefit import load_ct, do_fit


def _finite_mask(df: pd.DataFrame) -> np.ndarray:
    cols = ["k", "delta_ct2"]
    m = np.ones(len(df), dtype=bool)
    for c in cols:
        m &= np.isfinite(df[c].values)
    return m


def main():
    ap = argparse.ArgumentParser(description="Parameter sweep for slope-2 robustness (frequency-windowed).")
    ap.add_argument("--data", default="data/ct/ct_bounds.csv")
    ap.add_argument("--event", default=None)
    ap.add_argument("--fmin_list", default="30,40,60")
    ap.add_argument("--fmax_list", default="600,800,1024")
    ap.add_argument("--method_list", default="wls,huber")
    # 註記用途；實際回歸仍以 CSV 中現有點為準
    ap.add_argument("--n_bins_list", default="16,24,32")
    ap.add_argument("--min_points", type=int, default=8)
    args = ap.parse_args()

    df = load_ct(Path(args.data))
    if args.event:
        df = df[df["event"] == args.event]
    df = df[_finite_mask(df)].copy()

    # k = 2π f / c
    c = 299792458.0
    f2k = lambda f: 2 * np.pi * float(f) / c

    fmins = [float(x) for x in str(args.fmin_list).split(",")]
    fmaxs = [float(x) for x in str(args.fmax_list).split(",")]
    methods = [x.strip() for x in str(args.method_list).split(",") if x.strip()]
    n_bins = [int(x) for x in str(args.n_bins_list).split(",")]

    rows = []
    tmp = Path("data/ct/_tmp_rb.csv")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    for fm, fx, mth, nb in itertools.product(fmins, fmaxs, methods, n_bins):
        if fm >= fx:
            continue
        kmin, kmax = f2k(fm), f2k(fx)
        dff = df[(df["k"] >= kmin) & (df["k"] <= kmax) & _finite_mask(df)].copy()
        if len(dff) < args.min_points:
            continue
        dff.to_csv(tmp, index=False)
        r = do_fit(tmp, None, method=mth)  # 已用 f-窗口裁切，這裡不再套 profile
        n_used = int(getattr(r, "n_points", len(dff)))
        rows.append({
            "fmin": float(fm),
            "fmax": float(fx),
            "kmin": float(kmin),
            "kmax": float(kmax),
            "method": mth,
            "n_points": n_used,
            "slope": float(getattr(r, "slope")),
            "intercept_log10": float(getattr(r, "intercept_log10")),
            "n_bins_tag": int(nb),
        })

    Path("reports").mkdir(parents=True, exist_ok=True)
    outp = Path("reports/slope2_robustness.csv")
    pd.DataFrame(rows).to_csv(outp, index=False)
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
