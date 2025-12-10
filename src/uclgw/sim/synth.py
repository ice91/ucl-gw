# src/uclgw/sim/synth.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

C_LIGHT = 299_792_458.0  # m/s

@dataclass
class SynthConfig:
    event: str = "SIM_S2"
    ifos: List[str] = None
    fmin: float = 80.0
    fmax: float = 280.0
    n_bins: int = 20
    slope: float = 2.0          # 真值 s*
    log10A: float = -10.0       # 截距 in log10
    sigma_rel: float = 0.10     # 相對雜訊（log-domain 的 stdev 近似）
    hetero: bool = True         # True: 異方差 sigma ∝ y；False: 同方差
    p_out: float = 0.0          # 離群比例
    out_mult: float = 10.0      # 離群放大倍率
    seed: int = 42
    k0_hz: float = 100.0        # 正規化頻率 f0（定義 k0）

    def __post_init__(self):
        if self.ifos is None:
            self.ifos = ["H1", "L1", "V1"]

def _k_from_f(f_hz: np.ndarray) -> np.ndarray:
    # k = 2π f / c
    return 2.0 * np.pi * f_hz / C_LIGHT

def _freq_grid(fmin: float, fmax: float, n_bins: int) -> np.ndarray:
    # 用 logspace 模擬真實頻段的等比刻度
    return np.logspace(np.log10(max(1e-3, fmin)), np.log10(fmax), n_bins)

def synth_ct_bounds(cfg: SynthConfig) -> pd.DataFrame:
    """
    產生符合 ct_bounds 介面（event, ifo, f_hz, k, delta_ct2, sigma）的合成資料。
    模型：delta_ct2_true = 10^log10A * (k/k0)^slope
         觀測：log10(y_obs) = log10(y_true) + N(0, sigma_rel)
         離群：以 p_out 機率乘以 out_mult
         權重：hetero -> sigma = sigma_rel * y_obs；否則常數
    """
    rng = np.random.default_rng(cfg.seed)
    f = _freq_grid(cfg.fmin, cfg.fmax, cfg.n_bins)      # (n_bins,)
    k = _k_from_f(f)
    k0 = _k_from_f(np.array([cfg.k0_hz]))[0]
    # 真值（原尺度）
    y_true = (10.0 ** cfg.log10A) * (k / k0) ** cfg.slope

    # 在 log10 尺度加高斯噪聲 ⇒ 乘上 log-normal 因子
    logy_true = np.log10(np.clip(y_true, 1e-30, None))
    eps = rng.normal(loc=0.0, scale=cfg.sigma_rel, size=(cfg.n_bins,))  # log10 尺度
    logy_obs = logy_true + eps
    y_obs = 10.0 ** logy_obs

    # 離群
    if cfg.p_out > 0.0:
        m_out = rng.random(size=(cfg.n_bins,)) < cfg.p_out
        y_obs = np.where(m_out, y_obs * float(cfg.out_mult), y_obs)

    # 權重（sigma 代表原尺度標準差的 proxy；WLS 用 1/sigma^2）
    if cfg.hetero:
        sigma = np.maximum(cfg.sigma_rel * y_obs, 1e-30)
    else:
        sigma = np.full_like(y_obs, fill_value=max(cfg.sigma_rel * np.median(y_obs), 1e-30))

    rows = []
    for ifo in cfg.ifos:
        df_ifo = pd.DataFrame({
            "event": cfg.event,
            "ifo": ifo,
            "f_hz": f.astype(float),
            "k": k.astype(float),
            "delta_ct2": np.maximum(y_obs, 1e-30).astype(float),
            "sigma": sigma.astype(float),
        })
        rows.append(df_ifo)

    df = pd.concat(rows, ignore_index=True)
    return df
