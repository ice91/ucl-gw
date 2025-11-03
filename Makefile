
.PHONY: env repro-min baselines clean slope2 qa package

PY := python

env:
	@echo "Create/activate env per environment.yml (optional)."

repro-min:
	$(PY) -m scripts.flux_ratio_frw --routeX ROD --routeY CS --L 64 --grid 16
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
	$(PY) - <<'PYCODE'
import json, os, hashlib, shutil
root = "reports"
manifest = {"files": []}
if os.path.isdir(root):
    for fn in sorted(os.listdir(root)):
        p = os.path.join(root, fn)
        if os.path.isfile(p):
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            manifest["files"].append({"name": fn, "sha256": h})
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
shutil.make_archive("submission_envelope", "zip", ".")
print("Created submission_envelope.zip")
PYCODE

clean:
	rm -f reports/* figs/*
