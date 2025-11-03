
import argparse, json, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fit_powerlaw(x, y):
    lx = np.log10(x)
    ly = np.log10(y)
    s, b = np.polyfit(lx, ly, 1)   # ly = s*lx + b
    return s, b

def main():
    ap = argparse.ArgumentParser(description="NLO Slope-2 one-pager from ct_bounds.csv")
    ap.add_argument("--data", default="examples/ct_bounds.csv")
    ap.add_argument("--profile", default="configs/profiles/lisa_100Hz.yaml")
    ap.add_argument("--pivot-f", type=float, default=None)
    ap.add_argument("--outfile", default="reports/slope2.json")
    args = ap.parse_args()

    prof = {}
    if os.path.exists(args.profile):
        with open(args.profile, "r") as f:
            try:
                prof = json.load(f)
            except Exception:
                prof = {}
    kmin = float(prof.get("logfit", {}).get("min_k", 1e-13))
    kmax = float(prof.get("logfit", {}).get("max_k", 1e-11))

    df = pd.read_csv(args.data)
    if {"k", "delta_ct2"}.issubset(df.columns):
        k = df["k"].to_numpy()
        y = df["delta_ct2"].to_numpy()
    elif {"f_hz", "cT_upper"}.issubset(df.columns):
        c = 299792458.0
        f = df["f_hz"].to_numpy()
        k = 2*np.pi*f/c
        y = np.maximum(df["cT_upper"].to_numpy()**2 - 1.0, 1e-20)
    else:
        raise SystemExit("Unsupported columns; need (k, delta_ct2) or (f_hz, cT_upper).")

    sel = (k >= kmin) & (k <= kmax) & (y > 0)
    if sel.sum() < 3:
        raise SystemExit("Not enough points in the fitting window.")

    s_hat, b_hat = fit_powerlaw(k[sel], y[sel])
    accept = (s_hat >= 1.8) and (s_hat <= 2.2)

    os.makedirs("reports", exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    out = {
        "profile": args.profile,
        "data": args.data,
        "slope_hat": float(s_hat),
        "intercept_log10": float(b_hat),
        "window_k": [float(kmin), float(kmax)],
        "accept": bool(accept),
        "timestamp": time.time()
    }
    with open(args.outfile, "w") as f:
        json.dump(out, f, indent=2)

    md = (
        "# NLO Slope-2 One-Pager\n"
        f"Data: `{args.data}`\n"
        f"Profile: `{args.profile}`\n"
        f"Fit Window: k in [{kmin:.2e}, {kmax:.2e}]\n\n"
        f"- Estimated slope (s^): **{s_hat:.3f}**\n"
        f"- Intercept (log10 A): **{b_hat:.3f}**\n"
        f"- Verdict: **{'PASS' if accept else 'FAIL'}** (expect 2 +/- 0.2)\n\n"
        "See figure: `figs/nlo_slope_fit.png`\n"
    )
    with open("reports/nlo_onepager.md", "w") as f:
        f.write(md)

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(k[sel], y[sel])
    kline = np.logspace(np.log10(kmin), np.log10(kmax), 100)
    yline = (10**b_hat) * (kline**s_hat)
    ax.plot(kline, yline)
    ax.set_xlabel("k")
    ax.set_ylabel("delta c_T^2")
    ax.set_title(f"log-log fit slope = {s_hat:.3f}")
    fig.tight_layout()
    fig.savefig("figs/nlo_slope_fit.png", dpi=144)
    plt.close(fig)

    print(f"Wrote {args.outfile}, reports/nlo_onepager.md and figs/nlo_slope_fit.png ; PASS={accept}")

if __name__ == "__main__":
    main()
