.PHONY: env repro-min baselines clean slope2 qa package ci

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

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

slope2:
	$(PY) -m scripts.nlo_slope_fit --data data/ct/ct_bounds.csv --profile configs/profiles/lisa_100Hz.yaml

qa:
	$(MAKE) repro-min
	$(MAKE) baselines
	$(PY) -m scripts.qa_gate

package:
	# 產出 reports/manifest.json + submission_envelope.zip
	$(PY) -m scripts.package_manifest
	# 輸出環境指紋
	$(PIP) freeze > reports/pip-freeze.txt
	# 附上 configs 與 examples 的快照，方便審稿人比對
	zip -ur submission_envelope.zip configs examples reports figs README.md Makefile prediction_ledger.yaml

clean:
	rm -f reports/* figs/*

ci: qa package

