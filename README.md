# UCL-GW — Audit-Ready QA Kit for PT-Symmetric Quaternionic Spacetime (PTQ)

**Status:** working research code accompanying the “Operational Capstone (UCL)” memo.
**Scope:** operational/QA only (no new theory).
**Band-limited case study:** GW190412, 60–300 Hz, phase-fit estimator with coherence gates.

> **Conservative claim:** In 60–300 Hz for GW190412 we do **not** observe a robust (k^2) window; the log–log fitted slope (\hat{s}) deviates from (2). We therefore report a **fixed-slope** ((s{=}2)) **upper limit** (A_{\rm ul}) on (\delta c_T^2(k)=A,k^2). Result is **compatible with GR** ((c_T=1)). Conformance **Level-B**; Exit **INC** (inconclusive on NLO; governance PASS).

---

## 1) Posture & Non-Overlap

This code implements **operational** QA checks under the fixed PTQ posture used in the paper:

* PT-scalar projection for observables; projective invariance via a non-dynamical spurion (\epsilon).
* We certify only the **quadratic tensor sector** through three QA indicators:

  1. **Flux-Ratio** (route differences are improvement currents),
  2. **Tensor-Lock** (Ward-type (K=G) at quadratic order),
  3. **NLO slope** search for the constrained fingerprint (\delta c_T^2=A,k^2).

**Out of scope:** deriving PTQ from microphysics, formal proofs of C1–C3, and extended phenomenology. See the companion theory papers for those items.

---

## 2) Repository Layout (what each piece does)

```
configs/
  baselines/          # Minimal comparators used in Sec. 3.5 of the memo
    gr_flat.yaml
    horndeski_min.yaml
    dhost_ref.yaml
  profiles/           # Instrument/data profiles (e.g., LVK O3)
    lvk_o3.yaml
    lisa_100Hz.yaml
  tensor_specs.yaml   # Spec used by Tensor-Lock extraction

data/
  ct/                 # Phase-fit derived c_T dispersion points (examples + sims)
    ct_bounds.csv
    events/GW190412_ct_bounds.csv
    events/SIM_*.csv
  raw/gwosc/GW190412/ # Example GWOSC files (H1/L1/V1)
  work/               # PSDs, segments, whitened streams built by prep scripts

examples/             # Tiny CSV for smoke tests (ct_bounds.csv)
figs/                 # Auto plots (e.g., figs/nlo_slope_fit.png)
reports/              # Machine-readable outputs (JSON/CSV/MD) for audit
prediction_ledger.yaml# Ledger entries (hashes/timestamps/exit codes)

scripts/
  gw_fetch_gwosc.py         # (Optional) Fetch GWOSC H1/L1/V1 frames
  gw_prepare.py             # Build PSD/segments/whitened data
  gw_build_ct_bounds.py     # Build per-event ct_bounds from prepared streams
  coherence_scout.py        # Coherence metrics (gates) for the phase-fit
  nlo_slope_fit.py          # NLO log–log slope fit (+ one-pager & figure)
  slope2_perifo.py          # Per-IFO slope diagnostics
  slope2_null_perm.py       # Null permutations for robustness
  flux_ratio_frw.py         # Flux-Ratio (C2 audit)
  lock_check_tensor.py      # Tensor-Lock extraction (K,G) and Δ_lock
  ... (bootstrap, jackknife, robustness helpers)

src/uclgw/                 # Library code: I/O, preprocessing, features, eval, reporting
tests/                     # PyTest suite (smokes + end-to-end)
Makefile                   # Convenience targets (see below)
environment.yml            # Reproducible conda env
requirements.txt           # (pip alternative)
```

---

## 3) Installation

### Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate ucl-repro
```

### Option B — venv + pip (if you prefer)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Determinism:** scripts accept a `--seed` (default often fixed in code) and use double precision; fast-math disabled.

---

## 4) Quickstart (60 seconds)

Minimal smoke test on a tiny CSV:

```bash
python -m scripts.nlo_slope_fit \
  --ct-data examples/ct_bounds.csv \
  --profile configs/profiles/lvk_o3.yaml

# Outputs:
#   figs/nlo_slope_fit.png
#   reports/nlo_onepager.md
#   reports/slope2.json (summary)
#   reports/slope2_ci_m2.json (intervals)
#   reports/slope2_ul_m2.json (fixed-slope s=2 upper limit)
```

> Tip: any script supports `--help` for full options.

---

## 5) End-to-End (GW190412, 60–300 Hz, phase-fit)

**(A) Ingest (optional, if you do not already have data/raw/gwosc/GW190412):**

```bash
python -m scripts.gw_fetch_gwosc --event GW190412 --out data/raw/gwosc/GW190412
```

**(B) Prepare streams (PSDs, segments, whitened):**

```bash
python -m scripts.gw_prepare \
  --event GW190412 \
  --profile configs/profiles/lvk_o3.yaml \
  --out-prefix data/work
# Produces:
#   data/work/psd/GW190412_{H1,L1,V1}.csv
#   data/work/segments/GW190412.json
#   data/work/whitened/GW190412_{H1,L1,V1}.npz
```

**(C) Build phase-fit bounds for (\delta c_T^2) points:**

```bash
python -m scripts.gw_build_ct_bounds \
  --event GW190412 \
  --profile configs/profiles/lvk_o3.yaml \
  --out data/ct/events/GW190412_ct_bounds.csv
```

**(D) Coherence scouting (record gates used by the fit):**

```bash
python -m scripts.coherence_scout \
  --ct-data data/ct/events/GW190412_ct_bounds.csv \
  --profile configs/profiles/lvk_o3.yaml \
  --out reports/coherence_GW190412.json
```

**(E) NLO slope analysis (log–log fit) + one-pager:**

```bash
python -m scripts.nlo_slope_fit \
  --ct-data data/ct/events/GW190412_ct_bounds.csv \
  --profile configs/profiles/lvk_o3.yaml
# Produces:
#   figs/nlo_slope_fit.png
#   reports/nlo_onepager.md
#   reports/slope2.json
#   reports/slope2_ci_m2.json
#   reports/slope2_ul_m2.json
```

**(F) Per-IFO diagnostics and null permutations (robustness):**

```bash
python -m scripts.slope2_perifo \
  --ct-data data/ct/events/GW190412_ct_bounds.csv \
  --profile configs/profiles/lvk_o3.yaml \
  --out reports/slope2_perifo_GW190412_huber_uw.json

python -m scripts.slope2_null_perm \
  --ct-data data/ct/events/GW190412_ct_bounds.csv \
  --profile configs/profiles/lvk_o3.yaml \
  --out reports/slope2_null_perm_GW190412.json
```

**(G) Flux-Ratio (C2) and Tensor-Lock (C3) audits:**

```bash
python -m scripts.flux_ratio_frw \
  --modelX ROD --modelY CS+ \
  --profile configs/profiles/lvk_o3.yaml \
  --out reports/flux_ratio.json

python -m scripts.lock_check_tensor \
  --model-spec configs/tensor_specs.yaml \
  --out reports/lock_check.csv
```

**(H) Governance / Ledger:**
Artifacts and exit codes are summarized in `reports/` and appended to `prediction_ledger.yaml`. The GW190412 run in 60–300 Hz is classified:

* **Level:** B
* **Exit:** **INC** (inconclusive NLO; governance PASS)
* **Band-limited conclusion:** **compatible with GR** ((c_T=1)) and reported as a **fixed-slope** ((s{=}2)) **upper limit** (A_{\rm ul}).

---

## 6) Baselines (GR-flat / Horndeski-min / DHOST-ref)

Minimal comparators for sanity and “no-mimic” checks (posture-aligned assumptions, operational only):

```bash
# Example pattern (exact options: see --help)
python -m scripts.flux_ratio_frw      --baseline configs/baselines/gr_flat.yaml
python -m scripts.lock_check_tensor   --model-spec configs/tensor_specs.yaml
```

Your repository already includes baseline result files under `reports/` when available. When comparing bands or events, cite run IDs and hashes (see Manifest note below).

---

## 7) What the Output Files Mean (quick glossary)

| File                              | Meaning (operational)                                                                                                         |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `reports/nlo_onepager.md`         | One-page human-readable summary for the NLO fit (slope, CI, gate settings, pivot info, governance line).                      |
| `figs/nlo_slope_fit.png`          | Log–log fit and residuals for the current run.                                                                                |
| `reports/slope2.json`             | Summary of fitted slope (\hat{s}), pivot, RMS, gating metadata.                                                               |
| `reports/slope2_ci_m2.json`       | Confidence intervals and model notes for the slope fit.                                                                       |
| `reports/slope2_ul_m2.json`       | **Upper limit** (A_{\rm ul}) for (\delta c_T^2=A,k^2) **at fixed (s{=}2)** (used when result is INC or to report strict ULs). |
| `reports/coherence_*.json`        | Coherence “gates” actually used (min/wide/bin thresholds) and QC counters.                                                    |
| `reports/flux_ratio.json`         | Integrated boundary flux ratio and convergence history (C2 audit).                                                            |
| `reports/lock_check.csv`          | Extracted ((K,G)) and (\Delta_{\rm lock}) values (C3 audit).                                                                  |
| `reports/slope2_perifo_*.json`    | Per-IFO slope contributions (diagnostic only).                                                                                |
| `reports/slope2_null_perm_*.json` | Null permutations; distributional checks for robustness.                                                                      |
| `prediction_ledger.yaml`          | Timestamped, hash-keyed entries (artifact IDs, Level/Exit, profile) for reproducibility.                                      |

---

## 8) Conformance & Exit Codes

We use four standardized outcomes (reported in the one-pager and ledger):

* **PASS** — QA criteria met; posture checks pass.
* **FAIL** — Operational/numerical error (e.g., convergence or unit map).
* **OOP** — Out-of-posture (theory-level mismatch vs. assumptions).
* **INC** — Inconclusive (e.g., no robust (k^2) window in band).

**GW190412 (60–300 Hz) := Level-B / Exit INC**, with **fixed-slope** ((s{=}2)) **upper limit** (A_{\rm ul}) reported.

---

## 9) Reproducibility, Hashes, and Manifests

* Runs are **seeded**; numerical libraries pinned via `environment.yml`.
* Artifact hashes and run metadata are tracked in `prediction_ledger.yaml`.
* Use `scripts/package_manifest.py` (if provided in your workflow) to record file hashes and a manifest “snapshot” after a run.

---

## 10) Tests

A compact PyTest suite exercises I/O, conditioning, slope fits (sim & smoke), and an end-to-end path.

```bash
pytest -q
```

---

## 11) Citing the Papers

When publishing results obtained with this kit:

* **Operational QA, reporting, Repro-Kit** → cite the **Operational Capstone (this memo)**.
* **Formal C1–C3 results, posture math, Palatini structure** → cite **Paper II**.
* **Microphysical motivation (D-brane / heat-kernel hints)** → cite **Paper I**.

(Use the BibTeX keys suggested in the memo.)

---

## 12) Responsible Use (Band-Limited, No Detection Claim)

* Results are **band-specific** and **window-dependent**.
* “INC” is an acceptable, governance-compliant outcome.
* **Do not** interpret an upper limit as a positive detection.
* Cross-event accumulation should proceed via the fixed-slope (s{=}2) upper-limit table (per the memo’s Appendix/§7 guidance).

---

## 13) Makefile (if available)

Typical conveniences (run `make` to list):

```bash
make env          # Create conda env from environment.yml
make baselines    # Run minimal baseline comparators
make clean        # Remove generated figures/reports
```

---

## 14) Contact / Issues

Please open issues with:
(1) the exact command line,
(2) your `environment.yml` hash,
(3) the relevant entries from `prediction_ledger.yaml`, and
(4) the produced files under `reports/` and `figs/`.

---

**Summary:** This repository provides a conservative, audit-ready harness to evaluate dispersion **without** introducing new assumptions beyond the PTQ posture. For GW190412 (60–300 Hz), the NLO test is **INC** and we report a **fixed-slope (s{=}2) upper limit** (A_{\rm ul}), **compatible with GR** in this band.
