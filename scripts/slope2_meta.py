# scripts/slope2_meta.py
# Meta-analysis for slope-2 checks across GW events.
# - Supports single-event (fixed-effect only) with clear note.
# - DerSimonian–Laird random-effects when >=2 events.
# - Outputs CIs, heterogeneity (Q, I2, tau2) and per-event variances.

from __future__ import annotations
import argparse, json, sys, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from uclgw.eval.slopefit import load_ct, do_fit  # noqa: E402


def _safe_float(x, default=None):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _ci95(mean: float, se: float | None):
    if se is None:
        return None
    z = 1.959963984540054  # 2-sided 95%
    return [float(mean - z * se), float(mean + z * se)]


def main():
    ap = argparse.ArgumentParser(description="Fixed/Random-effects meta-analysis over events")
    ap.add_argument("--data", default="data/ct/ct_bounds.csv",
                    help="CSV with stacked ct-bounds (columns include: event,x,y,w,...)")
    ap.add_argument("--profile", default="configs/profiles/lvk_o3.yaml",
                    help="Profile file (kept for interface compatibility; not used here).")
    args = ap.parse_args()

    df = load_ct(Path(args.data))
    events = sorted(df["event"].unique())

    per = []
    warnings = []
    if len(events) == 0:
        print("No events found in data; nothing to do.")
        out = {
            "per_event": [],
            "fixed_effect_slope": None,
            "fixed_effect_se": None,
            "fixed_effect_ci95": None,
            "random_effect_slope": None,
            "random_effect_se": None,
            "random_effect_ci95": None,
            "Q": None,
            "I2_percent": None,
            "tau2": None,
            "n_events": 0,
            "warnings": ["No events present."]
        }
        Path("reports").mkdir(parents=True, exist_ok=True)
        Path("reports/slope2_meta.json").write_text(json.dumps(out, indent=2))
        print("Wrote reports/slope2_meta.json (empty)")
        return

    # Per-event WLS slope & variance estimate
    for ev in events:
        tmp = Path("data/ct/_tmp_meta.csv")
        dfe = df[df["event"] == ev].copy()
        if dfe.shape[0] < 3:
            warnings.append(f"[{ev}] too few points ({dfe.shape[0]}) for a stable fit.")
        dfe.to_csv(tmp, index=False)

        r = do_fit(tmp, None, method="wls")
        # Weighted residual variance -> slope variance via Sxx (weighted)
        x = dfe["x"].values
        y = dfe["y"].values
        w = dfe["w"].values

        # Weighted mean of x
        wx_sum = float(np.sum(w * x))
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            warnings.append(f"[{ev}] non-positive total weight; fallback to equal weights.")
            w = np.ones_like(w)
            w_sum = float(np.sum(w))
            wx_sum = float(np.sum(w * x))

        xbar_w = wx_sum / (w_sum + 1e-12)
        yhat = r.slope * x + r.intercept
        resid = y - yhat

        # Weighted residual variance (approx dof = sum(w) - 2)
        dof = max(w_sum - 2.0, 1.0)
        sigma2 = float(np.sum(w * resid**2) / (dof + 1e-12))

        # Weighted Sxx
        sxx = float(np.sum(w * (x - xbar_w) ** 2))
        if sxx <= 0:
            # Extremely pathological; give a large variance as a safeguard
            warnings.append(f"[{ev}] Sxx <= 0; assigning large variance.")
            var_beta = 1e6
        else:
            var_beta = sigma2 / (sxx + 1e-12)

        # Numerical floor to avoid inf weights downstream
        var_beta = max(float(var_beta), 1e-12)

        per.append({
            "event": ev,
            "n_points": int(dfe.shape[0]),
            "slope": float(r.slope),
            "intercept": float(r.intercept),
            "var": float(var_beta),
            "se": float(np.sqrt(var_beta)),
        })

    # If single-event: output fixed-effect = that event; random-effects N/A
    if len(per) == 1:
        s_fixed = per[0]["slope"]
        se_fixed = per[0]["se"]
        out = {
            "per_event": per,
            "fixed_effect_slope": float(s_fixed),
            "fixed_effect_se": float(se_fixed),
            "fixed_effect_ci95": _ci95(s_fixed, se_fixed),
            "random_effect_slope": None,
            "random_effect_se": None,
            "random_effect_ci95": None,
            "Q": None,
            "I2_percent": None,
            "tau2": None,
            "n_events": 1,
            "warnings": warnings + ["Random-effects meta not applicable for a single event."]
        }
        Path("reports").mkdir(parents=True, exist_ok=True)
        Path("reports/slope2_meta.json").write_text(json.dumps(out, indent=2))
        print("Wrote reports/slope2_meta.json (single-event mode)")
        return

    # >= 2 events: fixed- and random-effects (DerSimonian–Laird)
    vars_ = np.array([max(p["var"], 1e-12) for p in per], dtype=float)
    slopes = np.array([p["slope"] for p in per], dtype=float)
    w_iv = 1.0 / vars_  # inverse-variance weights

    sum_w = float(np.sum(w_iv))
    if sum_w <= 0:
        warnings.append("Non-positive total inverse-variance; falling back to equal weights.")
        w_iv = np.ones_like(w_iv)
        sum_w = float(np.sum(w_iv))

    # Fixed effect
    s_fixed = float(np.sum(w_iv * slopes) / sum_w)
    se_fixed = float(np.sqrt(1.0 / (sum_w + 1e-12)))

    # Heterogeneity Q
    Q = float(np.sum(w_iv * (slopes - s_fixed) ** 2))
    dfre = len(per) - 1
    I2 = None
    if Q > dfre and Q > 0:
        I2 = float(max((Q - dfre) / Q, 0.0) * 100.0)
    else:
        I2 = 0.0

    # DL tau^2
    C = float(sum_w - np.sum(w_iv ** 2) / (sum_w + 1e-12))
    if C <= 0:
        tau2 = 0.0
        warnings.append("C <= 0 in DL estimator; setting tau2=0 (reduces to fixed-effect).")
    else:
        tau2 = float(max((Q - dfre) / C, 0.0))

    # Random-effects weights: 1 / (var_i + tau2)
    w_star = 1.0 / (vars_ + tau2)
    sum_w_star = float(np.sum(w_star))
    if sum_w_star <= 0:
        warnings.append("Non-positive random-effects total weight; falling back to fixed-effect.")
        s_random = s_fixed
        se_random = se_fixed
    else:
        s_random = float(np.sum(w_star * slopes) / sum_w_star)
        se_random = float(np.sqrt(1.0 / (sum_w_star + 1e-12)))

    out = {
        "per_event": per,
        "fixed_effect_slope": float(s_fixed),
        "fixed_effect_se": float(se_fixed),
        "fixed_effect_ci95": _ci95(s_fixed, se_fixed),
        "random_effect_slope": float(s_random),
        "random_effect_se": float(se_random),
        "random_effect_ci95": _ci95(s_random, se_random),
        "Q": float(Q),
        "I2_percent": float(I2),
        "tau2": float(tau2),
        "n_events": int(len(per)),
        "warnings": warnings
    }

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/slope2_meta.json").write_text(json.dumps(out, indent=2))
    print("Wrote reports/slope2_meta.json")


if __name__ == "__main__":
    main()
