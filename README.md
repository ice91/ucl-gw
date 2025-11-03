
# ucl-gw (Phase 0 • Scaffold)

This repository provides a runnable scaffold for the UCL GW Repro‑Kit.
Phase 0 goals:
- minimal, deterministic environment (`environment.yml`)
- `make repro-min` runs three QA stubs: Flux-Ratio, Lock-Check, Slope-2
- `make baselines` runs lock-check on three frozen baselines (GR/Horndeski/DHOST).

## Quickstart
```bash
cd ucl-gw
# (optional) conda env create -f environment.yml && conda activate ucl-gw
make repro-min      # produces reports/* and figs/*
make baselines      # lock-check on three baseline models
```
Outputs:
- `reports/flux_ratio.json`
- `reports/lock_check.csv`
- `reports/slope2.json`, `reports/nlo_onepager.md`, `figs/nlo_slope_fit.png`

NOTE: All configs are JSON stored in `.yaml` files (YAML superset) to avoid external parsers in Phase 0.
