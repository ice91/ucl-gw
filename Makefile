.PHONY: env repro-min baselines clean slope2 qa package ci gw-fetch gw-prepare gw-all

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

EVENT    ?= GW170817
IFOS     ?= H1,L1,V1
FS       ?= 4096
DURATION ?= 32
SEED     ?= 7
NPERSEG  ?= 16384
NOVERLAP ?= 8192
GATE_Z   ?= 5.0

env:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

repro-min:
	$(PY) -m scripts.flux_ratio_frw --routeX ROD --routeY CS --L 64 --grid 32
	$(PY) -m scripts.lock_check_tensor --model configs/tensor_specs.yaml --tolerance 1e-6
	$(PY) -m scripts.nlo_slope_fit --data examples/ct_bounds.csv --profile configs/profiles/lisa_100Hz.yaml

baselines:
	$(PY) -m scripts.lock_check_tensor --model configs/baselines/gr_flat.yaml --tolerance 1e-6 --enforce-exit
	$(PY) -m scripts.lock_check_tensor --model configs/baselines/horndeski_min.yaml --tolerance 1e-6 --enforce-exit
	# dhost_ref 預期 FAIL，不加 --enforce-exit，讓 qa_gate 驗證它確實失敗
	$(PY) -m scripts.lock_check_tensor --model configs/baselines/dhost_ref.yaml --tolerance 1e-6

gw-fetch:
	$(PY) -m scripts.gw_fetch_gwosc --event $(EVENT) --ifos $(IFOS) --duration $(DURATION) --fs $(FS) --seed $(SEED)

gw-prepare:
	$(PY) -m scripts.gw_prepare --event $(EVENT) --nperseg $(NPERSEG) --noverlap $(NOVERLAP) --gate-z $(GATE_Z)

gw-all: gw-fetch gw-prepare

slope2:
	$(PY) -m scripts.nlo_slope_fit --data data/ct/ct_bounds.csv --profile configs/profiles/lvk_o3.yaml || \
	$(PY) -m scripts.nlo_slope_fit --data data/ct/ct_bounds.csv --profile configs/profiles/lisa_100Hz.yaml


qa:
	$(MAKE) repro-min
	$(MAKE) baselines
	$(PY) -m scripts.qa_gate

gw-ct:
	$(PY) -m scripts.gw_build_ct_bounds --event $(EVENT) --fmin 30 --fmax 1024 --n-bins 24 --mode proxy-k2 --aggregate

ct-aggregate:
	$(PY) -m scripts.gw_build_ct_bounds --event $(EVENT) --aggregate

gw-fetch-real:
	$(PY) -m scripts.gw_fetch_gwosc --event $(EVENT) --ifos $(IFOS) --duration $(DURATION) --fs $(FS) --download-gwosc

gw-all-real: gw-fetch-real gw-prepare

package:
	$(PY) -m scripts.package_manifest
	$(PIP) freeze > reports/pip-freeze.txt
	zip -ur submission_envelope.zip configs examples reports figs README.md Makefile prediction_ledger.yaml

clean:
	rm -f reports/* figs/*

ci: qa package
