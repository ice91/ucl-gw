# UCL-GW — Audit-Ready QA for PTQ / GR Tensor Dispersion

> **角色定位**
> 本庫是「PT-symmetric Quaternionic Spacetime（PTQ）」三部曲中的**操作層（Paper III / Capstone Memo）**。它把已在伴隨論文中建立的結構性結論，落實為**可稽核（audit-ready）**的三件套 QA 檢定與最小重現工具（Repro-Kit）。
> **不新增**任何理論假設、對稱性或自由度；在固定姿態（PT-scalar 投影＋projective invariance spurion (\epsilon)）下，僅做**可重現的檢定與上限匯報**。

---

## 0. TL;DR

```bash
# 0) 建環境
make env

# 1) 最小重現（三件套示範：Flux / Lock / NLO）
make repro-min

# 2) Baselines（GR-flat / Horndeski-min 應 PASS；DHOST-ref 預期 FAIL）
make baselines

# 3) 一鍵 QA Gate（彙整治理輸出）
make qa

# 4) 以 GW190412 產生 60–300 Hz 的 phase-fit 事件表並做 NLO 檢定（含 OFF）
make run-bbh190412
```

輸出重點會出現在：

* `reports/nlo_onepager.md`（**一頁報告**：(\hat s)、固定斜率上限 (A_{\rm ul})、Exit/Level）
* `figs/nlo_slope_fit.png`（對數域線性擬合＋殘差）
* `reports/slope2_ul_m2.json`（固定 (s{=}2) 的上限數值）
* `reports/flux_ratio.json`、`reports/lock_check.csv`（三件套治理證據）
* `prediction_ledger.yaml`（帳本條目與雜湊）
* `reports/aggregate_summary.json`（彙整摘要）

> **敘事原則（與論文一致）**：若在該頻帶**未見穩健 (k^2) 主導窗**，則 **NLO=INC**，並以**固定斜率 (s{=}2)** 回報係數上限 (A_{\rm ul})；此結果以 **「與 GR 相容」**（(c_T=1)）表述。

---

## 1. 三件套 QA 與對應腳本

| QA 指標                                    | 核心意涵（不超出論文）                                                                | 主要腳本 / 產物                                                                                                                                |
| ---------------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Flux-Ratio** (R_{X/Y}\to 1)            | 路徑差異僅為改良流 (\nabla_\mu J^\mu)（C2）                                           | `scripts/flux_ratio_frw.py` → `reports/flux_ratio.json`                                                                                  |
| **Tensor-Lock** (\Delta_{\rm lock}\to 0) | Ward-type locking (K=G)（C3）(\Rightarrow c_T) 侷限於光速於二次張量階                   | `scripts/lock_check_tensor.py` → `reports/lock_check.csv`                                                                                |
| **NLO slope**（尋找 (k^2) 指紋）               | 在允許窗中查驗 (\delta c_T^2(k)=A,k^2) 的**斜率 2**；若無穩健窗 (\Rightarrow) **INC + 上限** | `scripts/nlo_slope_fit.py` / `scripts/slope2_perifo.py` 等 → `figs/nlo_slope_fit.png`, `reports/slope2_*.json`, `reports/nlo_onepager.md` |

> **治理輸出**：`scripts/qa_gate.py` 會彙整為 `reports/aggregate_summary.json`，並根據檢定給出 **Exit code**（PASS/FAIL/OOP/INC）與**等級**（例：Level-B）。

---

## 2. 專案結構（精簡導覽）

```
configs/            # 設定檔：baselines/*、profiles/*、tensor_specs.yaml
data/
  ct/               # 聚合後的 c_T^2(k) 邊界資料（含事件明細）
  raw/gwosc/...     # 來自 GWOSC 的原始資料（H1/L1/V1）
  work/{psd,segments,whitened}/
examples/           # 範例資料（快速跑 NLO 示範）
figs/               # 繪圖輸出（如 nlo_slope_fit.png）
reports/            # 所有治理與數值報告（JSON/CSV/MD）
scripts/            # Repro-Kit CLI（見下）
src/uclgw/...       # 函式庫（io、features、eval、reporting、sim 等）
tests/              # pytest 測試
Makefile            # 一鍵目標（env, qa, phase-fit, baselines, package, …）
prediction_ledger.yaml  # 帳本（雜湊、版本、時間戳）
```

---

## 3. 安裝

```bash
make env
# 或自行：
# python3 -m venv .venv && .venv/bin/pip install -U pip -r requirements.txt
```

---

## 4. 最小重現／基準矩陣

### 4.1 最小重現

```bash
make repro-min
```

會依序產出：

* Flux：`reports/flux_ratio.json`
* Lock：`reports/lock_check.csv`
* NLO（範例資料）：`figs/nlo_slope_fit.png`、`reports/slope2_*.json`

### 4.2 Baselines（不做物理主張，只做工具鏈自檢）

```bash
make baselines
```

* `configs/baselines/gr_flat.yaml` → **PASS**（鎖定）
* `configs/baselines/horndeski_min.yaml` → **PASS**（鎖定）
* `configs/baselines/dhost_ref.yaml` → **預期 FAIL**（no-mimic 訊號）

> 這三格僅用於**工具鏈與治理**自我檢核；不宣稱物理新結論。

---

## 5. 真實事件：GW190412（60–300 Hz，phase-fit）

> 與論文 §5.4 的一頁報告敘事一致：本頻帶 **NLO=INC**，回報固定斜率 (s{=}2) 的 (A_{\rm ul})，並標記 **Level-B**；結果與 **GR 相容**。

### 5.1 一鍵流程（含 OFF）

```bash
make run-bbh190412
# 依指示先看 reports/coherence_GW190412.json 的 RECOMMEND，
# 對照 Makefile 中兩行 gw_build_ct_bounds 之門檻參數（ON / OFF）
```

完成後重點輸出：

* `reports/nlo_onepager.md`（包含 (\hat s)、(A_{\rm ul})、RMS、(f_\star)、Exit=INC、Level=B、hash）
* `figs/nlo_slope_fit.png`
* `reports/slope2_ul_m2.json`（固定 (s{=}2) 上限）
* `reports/coherence_GW190412.json`（相干偵查門檻）
* `reports/flux_ratio.json`、`reports/lock_check.csv`
* `prediction_ledger.yaml`（帳本條目＋雜湊）

> **敘事約束**：若 (\hat s\neq 2) 且沒有穩健 (k^2) 主導窗，該頻帶結論為 **INC**；僅回報固定斜率上限 **(A_{\rm ul})**，並以「與 GR 相容」描述。

---

## 6. 其他 Make 目標（快速檢索）

```bash
# 下載＋前處理（真實資料）
make gw-all-real EVENT=GW190412 IFOS=H1,L1,V1 DURATION=32 FS=4096

# 建立 phase-fit 事件表（primary: 30–800 Hz；stress: 30–1024 Hz）
make phase-fit-primary EVENT=GW170817
make phase-fit         EVENT=GW170817

# NLO 斜率（預設 profiles/lvk_o3.yaml；若失敗，回退到 lisa_100Hz.yaml）
make slope2

# 介面化的雙站/三站切分與權重法（per-IFO）
make perifo

# 穩健性／抽刀／排列（Null）檢定
make jackknife   EVENT=GW190412
make null-perm   EVENT=GW190412
make robustness  EVENT=GW190412
make null-block  EVENT=GW190412
make null-signflip EVENT=GW190412

# 一鍵 QA+Null 套餐（先建 primary，再跑 QA+Null）
make qa-plus2 EVENT=GW190412

# 打包審稿附件（含 pip-freeze、manifest、figs/reports）
make package
```

---

## 7. 設定檔與參數（審稿友善）

* `configs/profiles/*.yaml`
  頻帶、binning、是否丟棄 edge bin、coherence 門檻（如 `coh-wide-min`、`coh-min`、`coh-bin-min`）、加權法與不確定度傳播。

  > 建議：將實際使用之門檻寫入 profile，並在一頁報告中列示。

* `configs/baselines/*.yaml`
  **GR-flat**、**Horndeski-min**、**DHOST-ref** 的最小矩陣。

  > 語氣：工具鏈校驗用途；**不作物理聲明**。

* `configs/tensor_specs.yaml`
  Tensor kernel 規格，用於 `lock_check_tensor.py` 的 (K)/(G) 檢定。

* CLI 主要參數（節錄）

  * `scripts/gw_build_ct_bounds.py`：`--mode phase-fit|proxy-k2`、`--fmin`/`--fmax`、`--n-bins`、`--edges-mode`、`--drop-edge-bins`、`--gate-sec`、`--coh-wide-min`、`--coh-min`、`--coh-bin-min`、`--min-samples-per-bin`、`--label OFF`、`--null timeshift`、`--aggregate`
  * `scripts/nlo_slope_fit.py` / `scripts/slope2_perifo.py`：`--data`、`--profile`、`--method {wls,huber}`、`--preclean`、`--sigma-quantiles`、`--zmax`
  * Null/健壯性：`slope2_null_perm.py`（`--n-perm`、`--two-sided`）、`slope2_null_block.py`（`--block`）、`slope2_null_signflip.py`、`slope2_robustness.py`（多窗 sweep）

---

## 8. 輸出檔案對照（治理導向）

| 檔案                                                 | 內容                                                                         | 用途            |
| -------------------------------------------------- | -------------------------------------------------------------------------- | ------------- |
| `reports/nlo_onepager.md`                          | 事件、頻帶、(\hat s)、CI、固定 (s{=}2) 上限 (A_{\rm ul})、RMS、(f_\star)、Exit/Level、hash | **期刊附檔／審稿檢視** |
| `figs/nlo_slope_fit.png`                           | 對數域 (k)–(\delta c_T^2) 擬合、殘差、68% 帶                                         | 視覺審核          |
| `reports/slope2_ul_m2.json`                        | (A_{\rm ul})（固定 (s{=}2)）                                                   | 上限彙整          |
| `reports/flux_ratio.json`                          | 路徑通量比與誤差帶                                                                  | C2 稽核         |
| `reports/lock_check.csv`                           | (\Delta_{\rm lock}) 指標                                                     | C3 稽核         |
| `reports/coherence_*.json`                         | 相干門檻偵查建議值                                                                  | 盲調防護          |
| `reports/aggregate_summary.json`                   | 本次工作流彙總（包含 Exit/Level）                                                     | 一鍵總表          |
| `prediction_ledger.yaml`                           | 帳本條目（ID、hash、timestamp、seed）                                               | 可追溯／可稽核       |
| `reports/pip-freeze.txt`、`submission_envelope.zip` | 依 `make package` 生成                                                        | 審稿／封存         |

---

## 9. Exit Codes 與分級（與論文治理一致）

| 代碼       | 意義                                                            |
| -------- | ------------------------------------------------------------- |
| **PASS** | QA 條件皆滿足                                                      |
| **FAIL** | 操作／數值錯誤（例如規格不收斂、單位地圖漂移）                                       |
| **OOP**  | Out-of-Posture：超出既定姿態範圍（理論層級不相容）                              |
| **INC**  | Inconclusive：頻帶／窗型受系統雜訊或邊界效應支配，**固定 (s{=}2)** 回報 (A_{\rm ul}) |

> **GW190412（60–300 Hz）**：**INC**（未見穩健 (k^2) 窗），匯報 **固定 (s{=}2)** 之 (A_{\rm ul})，分級 **Level-B**；**與 GR 相容**。

---

## 10. 測試（CI／本地）

```bash
.venv/bin/pytest -q
# 或直接：
pytest -q
```

關鍵測試：

* `test_gwosc_io.py`（GWOSC 讀取）
* `test_conditioning.py`（前處理穩健）
* `test_dispersion_fit.py`、`test_slope2_end2end.py`（NLO 端到端）
* `test_realdata_smoke.py`（真實資料管線冒煙測試）
* `test_aggregate*.py`（彙整與路徑處理）

---

## 11. 常見紅旗與處置（對齊論文 §6）

* (R_{X/Y}) 非單調收斂或超出 ([1\pm\eta(h,L)]) → **邊界類別不符或姿態關閉**；修正 BC／啟用 PT-scalar。
* (\Delta_{\rm lock}) 對正規化敏感 → **單位地圖漂移或規格解析 bug**；鎖定單位地圖、以 `gr_flat` baseline 交叉檢核。
* (\hat s) 對 (f_\star)／資料 profile 敏感 → **系統雜訊主導**；改窗、啟用盲測、若持續則標記 **INC** 或 **OOP**（依姿態是否被破壞）。

---

## 12. 資料來源與重現性

* **資料來源**：`data/raw/gwosc/<EVENT>/{H1,L1,V1}.h5`
* **決定論**：所有隨機流程以 `--seed` 固定；`prediction_ledger.yaml` 記錄 hash／timestamp／版本；`make package` 會打包 `pip-freeze` 與主要產物，便於審稿封存與第三方重現。

---

## 13. 引用

```bibtex
@misc{Chen2025_capstone,    % 本庫與操作備忘錄（Paper III）
  title  = {Operational Capstone for a Unified Conservation Law ...},
  author = {Chien-Chih Chen}, year = {2025}, eprinttype={...}, eprint={...}
}
@article{Chen2025_luminality, % Paper II（結構定理與證明）
  title  = {Guaranteed Tensor Luminality from Symmetry ...},
  author = {Chien-Chih Chen}, year = {2025}
}
@article{Chen2025_string,     % Paper I（微觀動機）
  title  = {PT-Symmetric Quaternionic Spacetime from String Theory ...},
  author = {Chien-Chih Chen}, year = {2025}
}
```

> **用語統一**：在圖說／報告中，請使用 **“fixed-slope ((s{=}2)) upper limit”**、**“inconclusive (no (k^2)-dominated window)”**、**“compatible with GR ((c_T=1))”**，避免誤導為「已偵測」。

---

## 14. 授權與聯絡

* 授權：TBD
* 議題與回報：請以 `reports/aggregate_summary.json`、`prediction_ledger.yaml` 附上對應 hash／版本再開 issue，方便審核與追溯。

---

### 附錄：腳本速查（不改變任何理論內容）

* I/O 與前處理：`gw_fetch_gwosc.py`、`gw_prepare.py`、`io/gwosc.py`
* 事件表生成：`gw_build_ct_bounds.py`（`--mode phase-fit|proxy-k2`）
* NLO 家族：`nlo_slope_fit.py`、`slope2_perifo.py`、`slope2_jackknife.py`、`slope2_null_perm.py`、`slope2_null_block.py`、`slope2_null_signflip.py`、`slope2_robustness.py`、`slope2_meta.py`
* 三件套其餘：`flux_ratio_frw.py`、`lock_check_tensor.py`
* 相干偵查：`coherence_scout.py`
* 彙整與封裝：`qa_gate.py`、`package_manifest.py`


