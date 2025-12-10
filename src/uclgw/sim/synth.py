# src/uclgw/sim/synth.py
from dataclasses import dataclass
from typing import List, Union
import numpy as np
import pandas as pd

C_LIGHT = 299_792_458.0

@dataclass
class SynthConfig:
    event: str = "SIM"
    ifos: Union[str, List[str]] = ("H1", "L1", "V1")
    fmin: float = 80.0
    fmax: float = 280.0
    n_bins: int = 20
    k0_hz: float = 100.0          # 截距定義的參考頻率（只影響截距，不影響斜率）
    slope: float = 2.0
    log10A: float = -10.0
    sigma_rel: float = 0.10
    p_out: float = 0.0
    out_mult: float = 20.0
    hetero: int = 1               # 1: sigma ∝ y；0: 常數 sigma
    seed: int = 1

def _freq_grid(fmin: float, fmax: float, n: int) -> np.ndarray:
    return np.linspace(float(fmin), float(fmax), int(n))

def _k_from_f(f_hz: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * np.asarray(f_hz, dtype=float) / C_LIGHT

def synth_ct_bounds(cfg: SynthConfig) -> pd.DataFrame:
    if isinstance(cfg.ifos, str):
        ifos = [x.strip() for x in cfg.ifos.split(",") if x.strip()]
    else:
        ifos = list(cfg.ifos)

    f = _freq_grid(cfg.fmin, cfg.fmax, cfg.n_bins)
    k = _k_from_f(f)
    k0 = _k_from_f(np.array([cfg.k0_hz]))[0]

    y_true = (10.0 ** cfg.log10A) * (k / k0) ** cfg.slope
    logy_true = np.log10(np.clip(y_true, 1e-30, None))

    rows = []
    for j, ifo in enumerate(ifos):
        # 每個 IFO 使用不同 RNG，確保雜訊/離群獨立
        rng_ifo = np.random.default_rng(cfg.seed + 1000 + j)

        eps = rng_ifo.normal(loc=0.0, scale=cfg.sigma_rel, size=(cfg.n_bins,))
        logy_obs = logy_true + eps
        y_obs = 10.0 ** logy_obs

        if cfg.p_out > 0.0:
            m_out = rng_ifo.random(size=(cfg.n_bins,)) < cfg.p_out
            y_obs = np.where(m_out, y_obs * float(cfg.out_mult), y_obs)

        if cfg.hetero:
            sigma = np.maximum(cfg.sigma_rel * y_obs, 1e-30)
        else:
            sigma = np.full_like(y_obs, fill_value=max(cfg.sigma_rel * float(np.median(y_obs)), 1e-30))

        rows.append(pd.DataFrame({
            "event": cfg.event,
            "ifo": ifo,
            "f_hz": f.astype(float),
            "k": k.astype(float),
            "delta_ct2": np.maximum(y_obs, 1e-30).astype(float),
            "sigma": sigma.astype(float),
        }))

    return pd.concat(rows, ignore_index=True)
