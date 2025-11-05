# scripts/slope2_null_signflip.py
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from uclgw.eval.slopefit import do_fit, load_ct

def main():
    ap = argparse.ArgumentParser(description="Sign-flip residual null for slope-2")
    ap.add_argument("--data", default="data/ct/ct_bounds.csv")
    ap.add_argument("--profile", default="configs/profiles/lvk_o3.yaml")
    ap.add_argument("--event", default=None)
    ap.add_argument("--n-perm", type=int, default=5000)
    args = ap.parse_args()

    tmp = Path("data/ct/_tmp_signflip.csv")
    df = load_ct(Path(args.data))
    if args.event:
        df = df[df["event"] == args.event]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["delta_ct2"] > 0]
    df.to_csv(tmp, index=False)

    real = do_fit(tmp, args.profile, method="wls")
    x = real.x; y = real.y; w = real.w; yhat = real.yhat
    resid = y - yhat
    rng = np.random.default_rng(123)
    hits = 0; slopes = []
    X = np.vstack([x, np.ones_like(x)]).T
    W = np.diag(w); XtW = X.T @ W
    for _ in range(args.n_perm):
        signs = rng.choice([-1.0, 1.0], size=resid.size)
        yperm = yhat + resid * signs
        beta = np.linalg.inv(XtW @ X) @ (XtW @ yperm)
        slopes.append(float(beta[0]))
        if beta[0] >= real.slope:
            hits += 1
    p = (hits + 1) / (args.n_perm + 1)
    out = dict(slope_real=real.slope, perm_mean=float(np.mean(slopes)),
               perm_std=float(np.std(slopes, ddof=1)), p_value_one_sided=p,
               n_perm=args.n_perm)
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/slope2_null_signflip.json").write_text(json.dumps(out, indent=2))
    print("Wrote reports/slope2_null_signflip.json")

if __name__ == "__main__":
    main()
