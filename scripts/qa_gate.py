# scripts/qa_gate.py
import json, csv, sys, os

def fail(msg):
    print(f"[QA FAIL] {msg}")
    sys.exit(1)

def check_flux_ratio(path="reports/flux_ratio.json"):
    if not os.path.exists(path):
        fail(f"missing {path}")
    with open(path) as f:
        j = json.load(f)
    if not j.get("pass", False):
        fail("flux_ratio pass flag is False")
    print("[QA OK] flux_ratio")

def check_lock_check(path="reports/lock_check.csv"):
    if not os.path.exists(path):
        fail(f"missing {path}")
    # 至少要有 PASS，且 dhost-ref 必須 FAIL（no-mimic 控制）
    saw_pass = False
    saw_dhost_fail = False
    with open(path) as f:
        rd = csv.DictReader(f)
        rows = list(rd)
    if not rows:
        fail("lock_check.csv is empty")
    for r in rows:
        model = r.get("model","")
        passed = (r.get("pass","0") in ("1","True","true"))
        if passed: 
            saw_pass = True
        if model == "dhost-ref" and not passed:
            saw_dhost_fail = True
    if not saw_pass:
        fail("no PASS in lock_check.csv")
    if not saw_dhost_fail:
        fail("dhost-ref did not FAIL as expected")
    print("[QA OK] lock_check")

def check_slope2(path="reports/slope2.json"):
    if not os.path.exists(path):
        fail(f"missing {path}")
    with open(path) as f:
        j = json.load(f)
    if not j.get("accept", False):
        fail("slope2 accept flag is False")
    s = j.get("slope_hat", None)
    if s is None or not (1.8 <= float(s) <= 2.2):
        fail(f"slope_hat not in [1.8,2.2], got {s}")
    print(f"[QA OK] slope2 (ŝ={float(s):.3f})")

def main():
    check_flux_ratio()
    check_lock_check()
    check_slope2()
    print("[QA OK] all")

if __name__ == "__main__":
    main()
