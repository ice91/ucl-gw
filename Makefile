.PHONY: env repro-min baselines clean slope2 qa package ci gw-fetch gw-prepare gw-all scout bbh190412 bbh190814 bbh150914 bns170817_long run-bbh190412 run-bns170817

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

perifo:
	$(PY) -m scripts.slope2_perifo --data data/ct/ct_bounds.csv --profile configs/profiles/lvk_o3.yaml

jackknife:
	$(PY) -m scripts.slope2_jackknife --data data/ct/ct_bounds.csv --profile configs/profiles/lvk_o3.yaml --event $(EVENT) --n-bootstrap 400

null-perm:
	$(PY) -m scripts.slope2_null_perm --data data/ct/ct_bounds.csv --profile configs/profiles/lvk_o3.yaml --event $(EVENT) --n-perm 2000

robustness:
	$(PY) -m scripts.slope2_robustness --data data/ct/ct_bounds.csv --event $(EVENT) --fmin_list 30,40,60 --fmax_list 600,800,1024 --method_list wls,huber --n_bins_list 16,24,32

meta:
	$(PY) -m scripts.slope2_meta --data data/ct/ct_bounds.csv --profile configs/profiles/lvk_o3.yaml

qa-plus: qa perifo jackknife null-perm robustness meta

# 真・phase-fit 事件表（primary: 30–800 Hz；stress: 30–1024 Hz）
phase-fit:
	$(PY) -m scripts.gw_build_ct_bounds --event $(EVENT) --mode phase-fit --fmin 30 --fmax 1024 --n-bins 24 --aggregate

phase-fit-primary:
	$(PY) -m scripts.gw_build_ct_bounds --event $(EVENT) --mode phase-fit --fmin 30 --fmax 800 --n-bins 24 --aggregate

# Off-source/Null（以 timeshift 破壞跨站相干）
offsource:
	$(PY) -m scripts.gw_build_ct_bounds --event $(EVENT) --mode phase-fit --fmin 30 --fmax 800 --n-bins 24 --null timeshift --label OFF --aggregate

# 嚴格 Null 檢定
null-block:
	$(PY) -m scripts.slope2_null_block --data data/ct/ct_bounds.csv --profile configs/profiles/lvk_o3.yaml --event $(EVENT) --block 3 --n-perm 5000

null-signflip:
	$(PY) -m scripts.slope2_null_signflip --data data/ct/ct_bounds.csv --profile configs/profiles/lvk_o3.yaml --event $(EVENT) --n-perm 5000

# 強化版一鍵：先建 primary 事件表，再做一套 QA+Null
qa-plus2: phase-fit-primary slope2 perifo jackknife robustness null-block null-signflip

# 事件相干偵查（避免盲調）
scout:
	$(PY) -m scripts.coherence_scout --event $(EVENT) --fmin 60 --fmax 300 --nperseg 4096 --noverlap 3072 --gate-sec 0

# 下載＋前處理（真實資料）
bbh190412:
	$(MAKE) gw-fetch-real EVENT=GW190412 IFOS=H1,L1,V1 DURATION=32 FS=4096
	$(MAKE) gw-prepare    EVENT=GW190412 NPERSEG=8192 NOVERLAP=4096 GATE_Z=5.0

bbh190814:
	$(MAKE) gw-fetch-real EVENT=GW190814 IFOS=H1,L1,V1 DURATION=32 FS=4096
	$(MAKE) gw-prepare    EVENT=GW190814 NPERSEG=8192 NOVERLAP=4096 GATE_Z=5.0

bbh150914:
	$(MAKE) gw-fetch-real EVENT=GW150914 IFOS=H1,L1     DURATION=16 FS=4096
	$(MAKE) gw-prepare    EVENT=GW150914 NPERSEG=4096 NOVERLAP=2048 GATE_Z=5.0

# BNS：為了吃到前奏，拉長段長；之後 phase-fit 再用 gate-sec 決定實際視窗
bns170817_long:
	$(MAKE) gw-fetch-real EVENT=GW170817 IFOS=H1,L1,V1 DURATION=256 FS=4096
	$(MAKE) gw-prepare    EVENT=GW170817 NPERSEG=8192 NOVERLAP=4096 GATE_Z=5.0

# 一鍵範例：BBH190412（先 scout，拿到建議門檻後再跑 ON/OFF）
run-bbh190412: bbh190412
	$(PY) -m scripts.coherence_scout --event GW190412 --fmin 50 --fmax 300 --nperseg 1024 --noverlap 768 --gate-sec 1.0
	# ↑ 看 reports/coherence_GW190412.json 的 RECOMMEND，再把數值帶進下面兩行
	$(PY) -m scripts.gw_build_ct_bounds --event GW190412 --mode phase-fit --fmin 50 --fmax 300 --n-bins 8 \
		--gate-sec 1.0 --nperseg 1024 --noverlap 768 --edges-mode logspace \
		--coh-wide-min 0.25 --coh-min 0.00 --coh-bin-min 0.01 --min-samples-per-bin 2 --aggregate
	$(PY) -m scripts.gw_build_ct_bounds --event GW190412 --mode phase-fit --fmin 50 --fmax 300 --n-bins 8 \
		--gate-sec 1.0 --nperseg 1024 --noverlap 768 --edges-mode logspace \
		--coh-wide-min 0.25 --coh-min 0.00 --coh-bin-min 0.01 --min-samples-per-bin 2 \
		--null timeshift --label OFF --aggregate

# 一鍵範例：BNS GW170817（長 gate 吃到前奏）
run-bns170817: bns170817_long
	$(PY) -m scripts.coherence_scout --event GW170817 --fmin 60 --fmax 300 --nperseg 4096 --noverlap 3072 --gate-sec 40.0
	# ↑ 讀 RECOMMEND 後把數值帶入
	$(PY) -m scripts.gw_build_ct_bounds --event GW170817 --mode phase-fit --fmin 60 --fmax 300 --n-bins 10 \
		--gate-sec 40.0 --nperseg 4096 --noverlap 3072 --edges-mode logspace --drop-edge-bins 1 \
		--coh-wide-min 0.20 --coh-min 0.00 --coh-bin-min 0.008 --min-samples-per-bin 3 --aggregate
	$(PY) -m scripts.gw_build_ct_bounds --event GW170817 --mode phase-fit --fmin 60 --fmax 300 --n-bins 10 \
		--gate-sec 40.0 --nperseg 4096 --noverlap 3072 --edges-mode logspace --drop-edge-bins 1 \
		--coh-wide-min 0.20 --coh-min 0.00 --coh-bin-min 0.008 --min-samples-per-bin 3 \
		--null timeshift --label OFF --aggregate


package:
	$(PY) -m scripts.package_manifest
	$(PIP) freeze > reports/pip-freeze.txt
	zip -ur submission_envelope.zip configs examples reports figs README.md Makefile prediction_ledger.yaml

clean:
	rm -f reports/* figs/*

ci: qa package
