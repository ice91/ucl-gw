# scripts/slope2_meta.py
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from uclgw.eval.slopefit import load_ct, do_fit

def main():
    ap = argparse.ArgumentParser(description="Fixed/Random-effects meta-analysis over events")
    ap.add_argument("--data", default="data/ct/ct_bounds.csv")
    ap.add_argument("--profile", default="configs/profiles/lvk_o3.yaml")
    args = ap.parse_args()

    df = load_ct(Path(args.data))
    events = sorted(df["event"].unique().tolist())
    if not events:
        print("No events found.")
        return

    per = []
    for ev in events:
        tmp = Path("data/ct/_tmp_meta.csv")
        dfe = df[df["event"] == ev].copy()
        dfe.to_csv(tmp, index=False)
        r = do_fit(tmp, args.profile, method="wls")
        # 以殘差估計 slope variance（近似）
        X = np.vstack([r.x, np.ones_like(r.x)]).T
        yhat = X @ np.array([r.slope, r.intercept])
        resid = r.y - yhat
        sigma2 = float(np.sum(r.w * resid**2) / (np.sum(r.w) - 2.0 + 1e-9))
        sx2 = float(np.sum(r.w * (r.x - np.average(r.x, weights=r.w))**2))
        var_beta = sigma2 / (sx2 + 1e-12)
        per.append({"event": ev, "slope": float(r.slope), "var": float(var_beta)})

    # fixed-effects
    w = np.array([1.0/p["var"] for p in per])
    s = np.array([p["slope"] for p in per])
    s_fixed = float(np.sum(w*s)/np.sum(w))
    # random-effects (DerSimonian-Laird)
    Q = float(np.sum(w*(s - s_fixed)**2))
    dfre = max(len(per)-1, 1)
    C = float(np.sum(w) - np.sum(w**2)/np.sum(w))
    tau2 = max((Q - dfre)/C, 0.0)
    w_star = 1.0/(1.0/w + tau2)
    s_random = float(np.sum(w_star*s)/np.sum(w_star))

    out = {"per_event": per, "fixed_effect_slope": s_fixed,
           "random_effect_slope": s_random, "Q": Q, "tau2": tau2,
           "n_events": len(per)}
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/slope2_meta.json").write_text(json.dumps(out, indent=2))
    print("Wrote reports/slope2_meta.json ({} events)".format(len(per)))

if __name__ == "__main__":
    main()
