# scripts/qa_gate.py
from __future__ import annotations
import os, json, csv, sys
from typing import Tuple

FLUX_JSON = "reports/flux_ratio.json"
SLOPE_JSON = "reports/slope2.json"
LOCK_CSV  = "reports/lock_check.csv"

# 可調參（門檻）
SLOPE_MIN, SLOPE_MAX = 1.8, 2.2
REQUIRE_MODELS_PASS  = ["gr-flat", "horndeski-min"]
REQUIRE_MODELS_FAIL  = ["dhost", "dhost-ref", "DHOST"]

def _status(ok: bool, name: str, extra: str = ""):
    tag = "OK" if ok else "FAIL"
    msg = f"[QA {tag}] {name}"
    if extra:
        msg += f" {extra}"
    print(msg)
    if not ok:
        sys.exit(2)

def _check_flux() -> Tuple[bool, str]:
    if not os.path.isfile(FLUX_JSON):
        return False, "missing reports/flux_ratio.json"
    j = json.load(open(FLUX_JSON))
    ok = bool(j.get("PASS", j.get("pass", False)))
    return ok, ""

def _check_slope() -> Tuple[bool, str]:
    if not os.path.isfile(SLOPE_JSON):
        return False, "missing reports/slope2.json"
    j = json.load(open(SLOPE_JSON))
    if not bool(j.get("accept", False)):
        return False, "accept=False"
    s = float(j.get("slope_hat", 0.0))
    return (SLOPE_MIN <= s <= SLOPE_MAX), f"(ŝ={s:.3f})"

def _check_lock() -> Tuple[bool, str]:
    if not os.path.isfile(LOCK_CSV):
        return False, "missing reports/lock_check.csv"
    ok_pass = {k: False for k in REQUIRE_MODELS_PASS}
    ok_fail = {k: False for k in REQUIRE_MODELS_FAIL}
    with open(LOCK_CSV, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            name = row.get("name", "").strip()
            passed = row.get("PASS", row.get("pass", "")).strip().lower() in ("true", "1", "yes")
            for k in ok_pass:
                if k in name and passed:
                    ok_pass[k] = True
            for k in ok_fail:
                if k in name and (not passed):
                    ok_fail[k] = True
    ok = all(ok_pass.values()) and all(ok_fail.values())
    detail = f"pass={ok_pass} fail={ok_fail}"
    return ok, detail

def main():
    ok, extra = _check_flux()
    _status(ok, "flux_ratio", extra)

    ok, extra = _check_lock()
    _status(ok, "lock_check", extra)

    ok, extra = _check_slope()
    _status(ok, "slope2", extra)

    print("[QA OK] all")

if __name__ == "__main__":
    main()
