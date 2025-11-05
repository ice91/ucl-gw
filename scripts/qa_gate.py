# --- replace: scripts/qa_gate.py --- #
from __future__ import annotations
import os, json, csv, sys, re
from typing import Tuple

FLUX_JSON = "reports/flux_ratio.json"
SLOPE_JSON = "reports/slope2.json"
LOCK_CSV  = "reports/lock_check.csv"

SLOPE_MIN, SLOPE_MAX = 1.8, 2.2

# 允許多種名稱寫法（底線/連字號/大小寫/含路徑）
REQUIRE_PASS_TOKENS = [r"gr[-_]?flat", r"horndeski[-_]?min"]
REQUIRE_FAIL_TOKENS = [r"dhost", r"dhost[-_]?ref"]

def _status(ok: bool, name: str, extra: str = ""):
    tag = "OK" if ok else "FAIL"
    print(f"[QA {tag}] {name} {extra}")
    if not ok:
        sys.exit(2)

def _check_flux() -> Tuple[bool, str]:
    if not os.path.isfile(FLUX_JSON):
        return False, "missing reports/flux_ratio.json"
    j = json.load(open(FLUX_JSON))
    ok = bool(j.get("PASS", j.get("pass", j.get("ok", False))))
    return ok, ""

def _check_slope() -> Tuple[bool, str]:
    if not os.path.isfile(SLOPE_JSON):
        return False, "missing reports/slope2.json"
    j = json.load(open(SLOPE_JSON))
    if not bool(j.get("accept", j.get("PASS", j.get("ok", False)))):
        return False, "accept=False"
    s = float(j.get("slope_hat", 0.0))
    return (SLOPE_MIN <= s <= SLOPE_MAX), f"(ŝ={s:.3f})"

def _norm_name(s: str) -> str:
    s = s.strip().lower()
    s = s.split("/")[-1].split("\\")[-1]    # 去路徑
    s = s.replace(".yaml", "").replace(".yml", "")
    return s

def _parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("true","1","yes","y","pass","ok","accepted")

def _check_lock() -> Tuple[bool, str]:
    if not os.path.isfile(LOCK_CSV):
        return False, "missing reports/lock_check.csv"

    # 支援多種欄名
    name_keys = ("name","model","spec","file","path")
    pass_keys = ("PASS","pass","ok","result","accepted")

    seen_pass = {pat: False for pat in REQUIRE_PASS_TOKENS}
    seen_fail = {pat: False for pat in REQUIRE_FAIL_TOKENS}

    with open(LOCK_CSV, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # 找到名稱欄
            raw_name = None
            for k in name_keys:
                if k in row and row[k]:
                    raw_name = row[k]; break
            if not raw_name: 
                continue
            name = _norm_name(str(raw_name))

            # 找到布林欄
            raw_ok = None
            for k in pass_keys:
                if k in row:
                    raw_ok = row[k]; break
            ok = _parse_bool(raw_ok)

            # 規則：GR/Horndeski 必須 PASS；DHOST 必須 FAIL
            for pat in REQUIRE_PASS_TOKENS:
                if re.search(pat, name):
                    if ok: seen_pass[pat] = True
            for pat in REQUIRE_FAIL_TOKENS:
                if re.search(pat, name):
                    if not ok: seen_fail[pat] = True

    ok = all(seen_pass.values()) and all(seen_fail.values())
    detail = f"pass={seen_pass} fail={seen_fail}"
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
