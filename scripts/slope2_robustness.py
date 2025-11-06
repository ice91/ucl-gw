# scripts/slope2_robustness.py
import sys, argparse, itertools
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from uclgw.eval.slopefit import load_ct, do_fit

def _get_intercept_any(r):
    return float(getattr(r, "intercept_log10", getattr(r, "intercept", np.nan)))

def _get_n_used(r, fallback_len):
    for attr in ("mask_count","n_points","n"):
        v = getattr(r, attr, None)
        if v is not None: return int(v)
    return int(fallback_len)

def main():
    ap = argparse.ArgumentParser(description="Parameter sweep for robustness")
    ap.add_argument("--data", default="data/ct/ct_bounds.csv")
    ap.add_argument("--event", default=None)
    ap.add_argument("--fmin_list", default="30,40,60")
    ap.add_argument("--fmax_list", default="600,800,1024")
    ap.add_argument("--method_list", default="wls,huber")
    ap.add_argument("--n_bins_list", default="16,24,32")  # 只作標記
    args = ap.parse_args()

    df = load_ct(Path(args.data))
    if args.event:
        df = df[df["event"] == args.event]

    c = 299792458.0
    f2k = lambda f: 2*np.pi*float(f)/c

    fmins = [float(x) for x in args.fmin_list.split(",")]
    fmaxs = [float(x) for x in args.fmax_list.split(",")]
    methods = [x.strip() for x in args.method_list.split(",")]
    n_bins = [int(x) for x in args.n_bins_list.split(",")]

    rows = []
    for fm, fx, mth, nb in itertools.product(fmins, fmaxs, methods, n_bins):
        if fm >= fx: continue
        kmin, kmax = f2k(fm), f2k(fx)
        dff = df[(df["k"] >= kmin) & (df["k"] <= kmax)].copy()
        if len(dff) < 8:  # 太少點略過
            continue
        tmp = Path("data/ct/_tmp_rb.csv"); tmp.parent.mkdir(parents=True, exist_ok=True)
        dff.to_csv(tmp, index=False)
        r = do_fit(tmp, None, method=mth)
        rows.append({
            "fmin": fm, "fmax": fx, "kmin": float(kmin), "kmax": float(kmax),
            "method": mth, "n_points": _get_n_used(r, len(dff)),
            "slope": float(r.slope), "intercept_log10": _get_intercept_any(r),
            "n_bins_tag": nb
        })

    Path("reports").mkdir(parents=True, exist_ok=True)
    outp = Path("reports/slope2_robustness.csv")
    pd.DataFrame(rows).to_csv(outp, index=False)
    print(f"Wrote {outp}")

if __name__ == "__main__":
    main()
