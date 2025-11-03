# scripts/lock_check_tensor.py
import argparse, json, os, numpy as np, sys

def fro_norm(a):
    return float(np.linalg.norm(a, ord='fro'))

def main():
    ap = argparse.ArgumentParser(description="Lock-Check QA: compute Δlock = ||K-G||/||G|| (Frobenius).")
    ap.add_argument("--model", default="configs/tensor_specs.yaml", help="Path to JSON-in-YAML config")
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--seeds", nargs="*", default=["20250101"])
    ap.add_argument("--enforce-exit", action="store_true",
                    help="Exit with non-zero code if check fails")
    args = ap.parse_args()

    with open(args.model, "r") as f:
        spec = json.load(f)  # JSON content in .yaml file

    K = np.array(spec["kernels"]["K"], dtype=float)
    G = np.array(spec["kernels"]["G"], dtype=float)

    dlock = fro_norm(K - G) / max(1e-18, fro_norm(G))
    passed = dlock <= args.tolerance

    os.makedirs("reports", exist_ok=True)
    out_csv = os.path.join("reports", "lock_check.csv")
    header = not os.path.exists(out_csv)
    with open(out_csv, "a") as f:
        if header:
            f.write("model,delta_lock,tolerance,pass\n")
        f.write(f"{spec.get('model','custom')},{dlock:.6e},{args.tolerance:.1e},{int(passed)}\n")

    print(f"[{spec.get('model','custom')}] Δlock={dlock:.3e} tol={args.tolerance:.1e} PASS={passed}")

    # 修正這行：enforce_exit 用底線；加 getattr 保險
    if getattr(args, "enforce_exit", False) and not passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
