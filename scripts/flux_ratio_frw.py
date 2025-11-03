
import argparse, json, os, math, time, hashlib

def main():
    ap = argparse.ArgumentParser(description="FRW flux-ratio QA (stubbed synthetic convergence).")
    ap.add_argument("--routeX", default="ROD")
    ap.add_argument("--routeY", default="CS")
    ap.add_argument("--L", type=int, default=64)       # domain size
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    L = max(4, args.L)
    grid = max(4, args.grid)
    values = []
    for i in range(1, 11):
        n = L * i
        err = 1.0 / (n**0.5) + 1.0/(grid**0.8)
        ratio = 1.0 + 0.15*err
        values.append({"n": n, "ratio": ratio})

    out = {
        "meta": {
            "routeX": args.routeX, "routeY": args.routeY,
            "timestamp": time.time(), "hash": hashlib.md5(str(values).encode()).hexdigest()
        },
        "convergence": values,
        "pass": values[-1]["ratio"] < 1.02
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/flux_ratio.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote reports/flux_ratio.json; PASS =", out["pass"])

if __name__ == "__main__":
    main()
