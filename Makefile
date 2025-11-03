.PHONY: env repro-min baselines clean slope2 qa package

# Try python, fallback to python3
PY := $(shell command -v python 2>/dev/null || command -v python3 2>/dev/null)
ifeq ($(PY),)
$(error "No Python interpreter found on PATH. Please install python3.")
endif

env:
	@echo "Create/activate env per environment.yml (optional)."

repro-min:
	$(PY) -m scripts.flux_ratio_frw --routeX ROD --routeY CS --L 64 --grid 32
	$(PY) -m scripts.lock_check_tensor --model configs/tensor_specs.yaml --tolerance 1e-6
	$(PY) -m scripts.nlo_slope_fit --data examples/ct_bounds.csv --profile configs/profiles/lisa_100Hz.yaml

baselines:
	$(PY) -m scripts.lock_check_tensor --model configs/baselines/gr_flat.yaml --tolerance 1e-6
	$(PY) -m scripts.lock_check_tensor --model configs/baselines/horndeski_min.yaml --tolerance 1e-6
	$(PY) -m scripts.lock_check_tensor --model configs/baselines/dhost_ref.yaml --tolerance 1e-6

slope2:
	$(PY) -m scripts.nlo_slope_fit --data data/ct/ct_bounds.csv --profile configs/profiles/lisa_100Hz.yaml

qa: repro-min

package:
	$(PY) -m scripts.package_manifest

clean:
	rm -f reports/* figs/*
