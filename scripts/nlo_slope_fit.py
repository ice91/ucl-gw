import argparse, json, os, time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _maybe_load_profile(path):
    if not os.path.exists(path):
        return {}
    # 先試 JSON，再試 YAML（可選）
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        try:
            import yaml  # 可選；若未安裝則忽略
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

def fit_powerlaw_loglog(lx, ly, w=None):
    """
    對數域線性回歸：ly = s*lx + b
    若提供 w（= 1/sigma_log^2），則做加權最小平方法。
    """
    if w is None:
        s, b = np.polyfit(lx, ly, 1)
        return s, b
    # 加權線性回歸（對數域）
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

    # fallback：用 10%～90% 分位數做視窗；若仍不足，再退到全域
    qlo, qhi = np.quantile(k, [0.10, 0.90])
    sel = (k >= qlo) & (k <= qhi)
    if _count(sel) < min_points:
        qlo, qhi = np.min(k), np.max(k)
        sel = (k >= qlo) & (k <= qhi)

    return float(qlo), float(qhi), sel

def main():
    ap = argparse.ArgumentParser(description="NLO Slope-2 one-pager from ct_bounds.csv")
    ap.add_argument("--data", default="examples/ct_bounds.csv")
    ap.add_argument("--profile", default="configs/profiles/lisa_100Hz.yaml")
    ap.add_argument("--outfile", default="reports/slope2.json")
    args = ap.parse_args()

    prof = _maybe_load_profile(args.profile)

    df = pd.read_csv(args.data)
    # 兩種輸入介面：(k, delta_ct2[, sigma]) 或 (f_hz, cT_upper)
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

    # 擇窗（profile 指定，否則 fallback 到分位數）
    kmin, kmax, sel = choose_window_from_profile_or_data(k, prof, y=y, min_points_default=10)
    if sel.sum() < 3:
        raise SystemExit("Not enough points in the fitting window even after fallback.")

    # 對數域
    lx = np.log10(k[sel])
    ly = np.log10(y[sel])

    # 誤差加權（把 sigma_y 傳到 log10 y：sigma_log = sigma_y / (y * ln(10))）
    w = None
    if sigma is not None:
        sig = sigma[sel]
        good = (sig > 0) & (y[sel] > 0)
        lx, ly, sig = lx[good], ly[good], sig[good]
        if lx.size < 3:
            raise SystemExit("Not enough valid (k,y,sigma) points after filtering.")
        sigma_log = sig / (y[sel][good] * math.log(10.0))
        w = 1.0 / np.clip(sigma_log**2, 1e-30, None)

    # 回歸
    s_hat, b_hat = fit_powerlaw_loglog(lx, ly, w=w)
    accept = (s_hat >= 1.8) and (s_hat <= 2.2)

    os.makedirs("reports", exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    out = {
        "profile": args.profile,
        "data": args.data,
        "slope_hat": float(s_hat),
        "intercept_log10": float(b_hat),
        "window_k": [float(kmin), float(kmax)],
        "n_points": int(sel.sum()),
        "accept": bool(accept),
        "timestamp": time.time()
    }
    with open(args.outfile, "w") as f:
        json.dump(out, f, indent=2)

    # 圖
    fig, ax = plt.subplots()
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.scatter(k[sel], y[sel], label="used")
    # 擬合線
    kline = np.logspace(np.log10(kmin), np.log10(kmax), 100)
    yline = (10**b_hat) * (kline**s_hat)
    ax.plot(kline, yline, label=f"fit slope={s_hat:.3f}")
    ax.set_xlabel("k [1/m]")
    ax.set_ylabel("delta c_T^2")
    ax.set_title("NLO Slope-2 log-log fit")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figs/nlo_slope_fit.png", dpi=144)
    plt.close(fig)

    print(f"Wrote {args.outfile}, reports/nlo_onepager.md and figs/nlo_slope_fit.png ; PASS={accept}")

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

if __name__ == "__main__":
    main()
