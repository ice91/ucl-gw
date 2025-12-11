# scripts/nlo_slope_fit.py
import argparse, json, os, time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# 允許 YAML/JSON profile
def _maybe_load_profile(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        try:
            import yaml
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

def fit_powerlaw_loglog(lx, ly, w=None):
    """
    ly = s*lx + b in log10 domain.
    w = 1/sigma_log^2
    """
    if w is None:
        s, b = np.polyfit(lx, ly, 1)
        return s, b
    W = np.sqrt(np.clip(w, 1e-30, None))
    s, b = np.polyfit(lx, ly, 1, w=W)
    return s, b

def choose_window_from_profile_or_data(k, prof, y=None, min_points_default=3):
    logfit = (prof.get("logfit") or {})
    kmin = logfit.get("min_k", None)
    kmax = logfit.get("max_k", None)
    min_points = int(logfit.get("min_points", min_points_default))

    def _count(sel): return int(np.count_nonzero(sel))

    if kmin is not None and kmax is not None and kmax > kmin:
        sel = (k >= float(kmin)) & (k <= float(kmax))
        if _count(sel) >= min_points:
            return float(kmin), float(kmax), sel

    # fallback：10%～90% 分位
    qlo, qhi = np.quantile(k, [0.10, 0.90])
    sel = (k >= qlo) & (k <= qhi)
    if _count(sel) < min_points:
        qlo, qhi = np.min(k), np.max(k)
        sel = (k >= qlo) & (k <= qhi)

    return float(qlo), float(qhi), sel

def _norm_ppf(p: float) -> float:
    # Acklam approximation
    a = [ -3.969683028665376e+01,  2.209460984245205e+02,
          -2.759285104469687e+02,  1.383577518672690e+02,
          -3.066479806614716e+01,  2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,  1.615858368580409e+02,
          -1.556989798598866e+02,  6.680131188771972e+01,
          -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03, -3.223964580411365e-01,
          -2.400758277161838e+00, -2.549732539343734e+00,
           4.374664141464968e+00,  2.938163982698783e+00 ]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00 ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        num = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
        den = ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        return num/den
    if phigh < p:
        q = math.sqrt(-2*math.log(1-p))
        num = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
        den = ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        return num/den
    q = p-0.5
    r = q*q
    num = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q
    den = (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    return num/den

def _prepare_from_csv(path_csv: str):
    df = pd.read_csv(path_csv)
    if {"k", "delta_ct2"}.issubset(df.columns):
        k = df["k"].to_numpy()
        y = df["delta_ct2"].to_numpy()
        sigma = df["sigma"].to_numpy() if "sigma" in df.columns else None
    elif {"f_hz", "cT_upper"}.issubset(df.columns):
        c = 299792458.0
        f = df["f_hz"].to_numpy()
        k = 2*np.pi*f/c
        y = np.maximum(df["cT_upper"].to_numpy()**2 - 1.0, 1e-30)
        sigma = None
    else:
        raise SystemExit("Unsupported columns; need (k, delta_ct2) or (f_hz, cT_upper).")
    return k, y, sigma

def _to_log_domain(k, y, sigma):
    lx = np.log10(k)
    ly = np.log10(np.maximum(y, 1e-30))
    w = None
    if sigma is not None:
        ln10 = math.log(10.0)
        sigma_log = sigma / (np.maximum(y, 1e-30) * ln10)
        w = 1.0 / np.clip(sigma_log**2, 1e-30, None)
    return lx, ly, w

def main():
    ap = argparse.ArgumentParser(description="NLO slope fit and fixed-slope UL from ct_bounds.csv")
    ap.add_argument("--data", default="examples/ct_bounds.csv")
    ap.add_argument("--profile", default="configs/profiles/lisa_100Hz.yaml")
    ap.add_argument("--outfile", default="reports/slope2.json")
    ap.add_argument("--fix-slope", type=float, default=None,
                    help="If set, fit intercept only with this fixed slope m, and report A & upper limit.")
    ap.add_argument("--alpha", type=float, default=0.05, help="One-sided alpha for UL (default 0.05).")
    ap.add_argument("--two-sided", action="store_true",
                    help="If set with --fix-slope, report two-sided (1-alpha) bound on intercept; default one-sided UL.")
    args = ap.parse_args()

    prof = _maybe_load_profile(args.profile)
    k, y, sigma = _prepare_from_csv(args.data)

    # 擇窗
    kmin, kmax, sel = choose_window_from_profile_or_data(k, prof, y=y, min_points_default=10)
    if sel.sum() < 3:
        raise SystemExit("Not enough points in the fitting window even after fallback.")

    k_sel, y_sel = k[sel], y[sel]
    lx, ly, w = _to_log_domain(k_sel, y_sel, sigma[sel] if sigma is not None else None)

    os.makedirs("reports", exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    out = {
        "profile": args.profile,
        "data": args.data,
        "window_k": [float(kmin), float(kmax)],
        "n_points": int(sel.sum()),
        "timestamp": time.time()
    }

    # ===== 分支 1：自由斜率（原本流程） =====
    if args.fix_slope is None:
        s_hat, b_hat = fit_powerlaw_loglog(lx, ly, w=w)
        accept = (s_hat >= 1.8) and (s_hat <= 2.2)
        out.update({
            "slope_hat": float(s_hat),
            "intercept_log10": float(b_hat),
            "accept": bool(accept)
        })

        # 畫圖
        fig, ax = plt.subplots()
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.scatter(k_sel, y_sel, label="used")
        kline = np.logspace(np.log10(kmin), np.log10(kmax), 200)
        yline = (10**b_hat) * (kline**s_hat)
        ax.plot(kline, yline, label=f"fit slope={s_hat:.3f}")
        ax.set_xlabel("k [1/m]")
        ax.set_ylabel("delta c_T^2")
        ax.set_title("NLO Slope-2 log-log fit")
        ax.legend()
        fig.tight_layout()
        fig.savefig("figs/nlo_slope_fit.png", dpi=144)
        plt.close(fig)

        with open(args.outfile, "w") as f:
            json.dump(out, f, indent=2)

        # one-pager
        md = (
            "# NLO Slope-2 One-Pager\n"
            f"Data: `{args.data}`\n"
            f"Profile: `{args.profile}`\n"
            f"Fit Window: k in [{kmin:.2e}, {kmax:.2e}] (points={out['n_points']})\n\n"
            f"- Estimated slope (s^): **{s_hat:.3f}**\n"
            f"- Intercept (log10 A): **{b_hat:.3f}**\n"
            f"- Verdict: **{'PASS' if accept else 'FAIL'}** (expect 2 ± 0.2)\n\n"
            "See figure: `figs/nlo_slope_fit.png`\n"
        )
        with open("reports/nlo_onepager.md", "w") as f:
            f.write(md)

        print(f"Wrote {args.outfile}, reports/nlo_onepager.md and figs/nlo_slope_fit.png ; PASS={accept}")
        return

    # ===== 分支 2：固定斜率，估截距與上限 =====
    m_fixed = float(args.fix_slope)
    wx = np.clip(w if w is not None else np.ones_like(lx), 1e-30, None)

    num = np.sum(wx * (ly - m_fixed * lx))
    den = np.sum(wx)
    b_hat = num / den

    # 殘差與 SE（sandwich 調整）
    r = ly - (m_fixed * lx + b_hat)
    s2 = np.sum(wx * r * r) / max((lx.size - 1), 1)  # 調整殘差
    se_known = math.sqrt(1.0 / den)
    se_b = max(se_known, math.sqrt(s2 / den))

    if args.two_sided:
        # two-sided (1-alpha) CI：b_hat ± z_{1 - alpha/2} se
        z = _norm_ppf(1.0 - args.alpha/2.0)
        b_lo = b_hat - z * se_b
        b_hi = b_hat + z * se_b
        A_lo = 10.0**b_lo
        A_hi = 10.0**b_hi
        A_ul = A_hi  # 報上側
    else:
        # one-sided UL：b_ul = b_hat + z_{1-alpha} se
        z = _norm_ppf(1.0 - args.alpha)
        b_lo = None
        b_hi = b_hat + z * se_b
        A_lo = None
        A_hi = 10.0**b_hi
        A_ul = A_hi

    A_hat = 10.0**b_hat

    out.update({
        "slope_fixed": m_fixed,
        "intercept_log10": float(b_hat),
        "intercept_se": float(se_b),
        "A_hat": float(A_hat),
        "A_ul": float(A_ul),
        "alpha": float(args.alpha),
        "two_sided": bool(args.two_sided)
    })

    with open(args.outfile, "w") as f:
        json.dump(out, f, indent=2)

    # 圖：用固定斜率畫線
    fig, ax = plt.subplots()
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.scatter(k_sel, y_sel, label="used")
    kline = np.logspace(np.log10(kmin), np.log10(kmax), 200)
    y_hat_line = (10**b_hat) * (kline**m_fixed)
    ax.plot(kline, y_hat_line, label=f"fixed m={m_fixed:g}, A={A_hat:.2e}")
    if not args.two_sided:
        y_ul_line = (10**(b_hat + _norm_ppf(1.0 - args.alpha) * se_b)) * (kline**m_fixed)
        ax.plot(kline, y_ul_line, linestyle="--", label=f"{int((1-args.alpha)*100)}% UL")
    else:
        z = _norm_ppf(1.0 - args.alpha/2.0)
        y_hi = (10**(b_hat + z*se_b)) * (kline**m_fixed)
        y_lo = (10**(b_hat - z*se_b)) * (kline**m_fixed)
        ax.plot(kline, y_hi, linestyle="--", label=f"{int((1-args.alpha)*100)}% upper")
        ax.plot(kline, y_lo, linestyle="--", label=f"{int((1-args.alpha)*100)}% lower")
    ax.set_xlabel("k [1/m]")
    ax.set_ylabel("delta c_T^2")
    ax.set_title("Fixed-slope power-law fit (log-log)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figs/nlo_slope_fit.png", dpi=144)
    plt.close(fig)

    # one-pager（上限版）
    ci_str = (f"[A_lo={A_lo:.2e}, A_hi={A_ul:.2e}]" if args.two_sided
              else f"UL (one-sided {int((1-args.alpha)*100)}%): A < {A_ul:.2e}")
    md = (
        "# Fixed-Slope Power-Law (Upper Limit)\n"
        f"Data: `{args.data}`\n"
        f"Profile: `{args.profile}`\n"
        f"Fit Window: k in [{kmin:.2e}, {kmax:.2e}] (points={out['n_points']})\n\n"
        f"- Fixed slope m: **{m_fixed:g}**\n"
        f"- Intercept (log10 A): **{b_hat:.3f}**  (SE **{se_b:.3f}**)\n"
        f"- A_hat: **{A_hat:.2e}**\n"
        f"- {ci_str}\n\n"
        "See figure: `figs/nlo_slope_fit.png`\n"
    )
    with open("reports/nlo_onepager.md", "w") as f:
        f.write(md)

    print(f"Wrote {args.outfile}, reports/nlo_onepager.md and figs/nlo_slope_fit.png ; FIXED_SLOPE={m_fixed}")
    return

if __name__ == "__main__":
    main()
