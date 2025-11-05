# scripts/slope2_jackknife.py
import argparse, json, sys, random
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from uclgw.eval.slopefit import load_ct, do_fit, window_mask, _read_profile_window

def main():
    ap = argparse.ArgumentParser(description="Jackknife / Bootstrap stability for slope-2")
    ap.add_argument("--data", default="data/ct/ct_bounds.csv")
    ap.add_argument("--profile", default="configs/profiles/lvk_o3.yaml")
    ap.add_argument("--event", default=None)
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    df = load_ct(Path(args.data))
    if args.event:
        df = df[df["event"] == args.event]
    kmin, kmax = _read_profile_window(Path(args.profile))
    m = window_mask(df, kmin, kmax)
    dfw = df[m].reset_index(drop=True)

    # Leave-one-out jackknife
    jk = []
    for i in range(len(dfw)):
        mask = np.ones(len(dfw), dtype=bool); mask[i] = False
        tmp_path = Path("data/ct/_tmp_jk.csv")
        dfw[mask].to_csv(tmp_path, index=False)
        r = do_fit(tmp_path, None, method="wls")
        jk.append(r.slope)
    jk = np.array(jk)
    jk_mean = jk.mean()
    jk_se = np.sqrt((len(jk)-1)/len(jk) * np.sum((jk - jk_mean)**2))

    # Bootstrap
    bs = []
    for _ in range(args.n_bootstrap):
        idx = np_rng.integers(0, len(dfw), len(dfw))
        tmp_path = Path("data/ct/_tmp_bs.csv")
        dfw.iloc[idx].to_csv(tmp_path, index=False)
        r = do_fit(tmp_path, None, method="wls")
        bs.append(r.slope)
    bs = np.array(bs)
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

    out = {
        "n_points": int(len(dfw)),
        "jk_mean": float(jk_mean),
        "jk_se": float(jk_se),
        "bootstrap_ci95": ci,
        "bootstrap_n": args.n_bootstrap
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/slope2_jackknife.json").write_text(json.dumps(out, indent=2))
    print("Wrote reports/slope2_jackknife.json")

if __name__ == "__main__":
    main()
