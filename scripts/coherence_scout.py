# scripts/coherence_scout.py
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from uclgw.features.dispersion_fit import _load_whitened, _welch_csd_xy  # 直接沿用既有實作

def _gate_intersection(W: dict, fs_ref: float, gate_sec: float):
    """與 phasefit_points 一致的 gating（交集窗）；gate_sec=0 時不動作。"""
    if not gate_sec or gate_sec <= 0:
        return W
    t0_idx = []
    for ifo in W:
        w = np.asarray(W[ifo]["w"])
        t0_idx.append(int(W[ifo]["meta"].get("t0_idx", int(np.argmax(np.abs(w))))))
    rad = int(max(1, gate_sec * fs_ref))
    lo_cands, hi_cands = [], []
    for ifo in W:
        w = np.asarray(W[ifo]["w"])
        t0 = int(W[ifo]["meta"].get("t0_idx", int(np.argmax(np.abs(w)))))
        lo_cands.append(max(0, t0 - rad))
        hi_cands.append(min(w.size, t0 + rad))
    lo, hi = int(max(lo_cands)), int(min(hi_cands))
    if hi - lo < 16:
        return W  # 太短，維持原樣
    for ifo in W:
        w = np.asarray(W[ifo]["w"]); t = np.asarray(W[ifo]["t"])
        W[ifo]["w"] = w[lo:hi]; W[ifo]["t"] = t[lo:hi]
    return W

def _pairs(keys):
    keys = sorted(keys); out = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            out.append((keys[i], keys[j]))
    return out

def _welch_m(n: int, nperseg: int, noverlap: int | None) -> int:
    if noverlap is None: noverlap = nperseg // 2
    step = max(1, nperseg - noverlap)
    if nperseg > n: return 1
    return 1 + max(0, (n - nperseg) // step)

def main():
    ap = argparse.ArgumentParser(description="Coherence scout (quantiles & segment counts) for planning phase-fit.")
    ap.add_argument("--event", required=True)
    ap.add_argument("--fmin", type=float, default=60.0)
    ap.add_argument("--fmax", type=float, default=300.0)
    ap.add_argument("--nperseg", type=int, default=4096)
    ap.add_argument("--noverlap", type=int, default=None)
    ap.add_argument("--gate-sec", type=float, default=0.0)
    args = ap.parse_args()

    W = _load_whitened(ROOT / "data/work/whitened", args.event)
    ifos = sorted(W.keys())
    fs_ref = float(W[ifos[0]]["meta"]["fs"])
    W = _gate_intersection(W, fs_ref, args.gate_sec)

    recs = []
    q_levels = [0.05, 0.10, 0.20, 0.30, 0.60, 0.75]

    for (a, b) in _pairs(ifos):
        x = W[a]["w"]; y = W[b]["w"]
        f, Cxy, Sxx, Syy = _welch_csd_xy(x, y, fs=fs_ref, nperseg=args.nperseg, noverlap=args.noverlap)
        coh2 = np.clip((np.abs(Cxy)**2) / (np.maximum(Sxx,1e-30)*np.maximum(Syy,1e-30)), 0.0, 1.0)
        band = (f >= args.fmin) & (f <= args.fmax)
        c = np.asarray(coh2[band], dtype=float)
        if c.size == 0:
            quants = {f"P{int(q*100)}": 0.0 for q in q_levels}
        else:
            quants = {f"P{int(q*100)}": float(np.quantile(c, q)) for q in q_levels}
        m = _welch_m(min(x.size, y.size), args.nperseg, args.noverlap)
        recs.append({"pair": f"{a}-{b}", "m": m, "quantiles": quants})

    # 建議門檻（跨 pair 聚合）
    def agg(name): return [r["quantiles"][name] for r in recs]
    P10 = agg("P10"); P20 = agg("P20"); P30 = agg("P30"); P60 = agg("P60"); P75 = agg("P75")
    rec = {
        "coh_min": round(max(0.0, min(P10)), 4),
        "coh_bin_min": round(float(np.median(P20 if np.median(P20)>0 else P30)), 4),
        "coh_wide_min": round(float(np.median(P60 if np.median(P60)>0 else P75)), 3),
        "min_samples_per_bin": 2,
        "edges_mode": "logspace"
    }
    out = {"event": args.event, "fmin": args.fmin, "fmax": args.fmax,
           "nperseg": args.nperseg, "noverlap": args.noverlap,
           "gate_sec": args.gate_sec, "pairs": recs,
           "m_min": int(min([r["m"] for r in recs])), "recommend": rec}

    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    path = ROOT / f"reports/coherence_{args.event}.json"
    path.write_text(json.dumps(out, indent=2))
    # 簡表輸出
    print(f"Wrote {path}")
    print("PAIR  m  P10   P20   P30   P60   P75")
    for r in recs:
        q = r["quantiles"]
        print(f"{r['pair']:6s} {r['m']:2d} {q['P10']:.4f} {q['P20']:.4f} {q['P30']:.4f} {q['P60']:.4f} {q['P75']:.4f}")
    print("RECOMMEND:", rec)

if __name__ == "__main__":
    main()
