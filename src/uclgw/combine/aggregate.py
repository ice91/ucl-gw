# src/uclgw/combine/aggregate.py
from __future__ import annotations
import os, glob, math, json
from dataclasses import dataclass, asdict
from typing import List, Dict
import pandas as pd

REQUIRED_COLS = ["event", "ifo", "f_hz", "k", "delta_ct2", "sigma"]

@dataclass
class AggregateSummary:
    out_csv: str
    n_files: int
    n_rows: int
    n_events: int
    events: List[str]

def _is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False

def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event"] = df["event"].astype(str)
    df["ifo"] = df["ifo"].astype(str)
    for c in ["f_hz", "k", "delta_ct2", "sigma"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_event_csvs(events_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(events_dir, "*_ct_bounds.csv")))

def aggregate(events_dir: str = "data/ct/events",
              out_csv: str = "data/ct/ct_bounds.csv",
              write_summary_json: str | None = None) -> AggregateSummary:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fns = load_event_csvs(events_dir)
    if not fns:
        # 建立空框架以利 downstream 腳本處理
        pd.DataFrame(columns=REQUIRED_COLS).to_csv(out_csv, index=False)
        summary = AggregateSummary(out_csv, 0, 0, 0, [])
        if write_summary_json:
            with open(write_summary_json, "w") as f:
                json.dump(asdict(summary), f, indent=2)
        return summary

    frames: List[pd.DataFrame] = []
    for fn in fns:
        df = pd.read_csv(fn)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{fn} 缺少欄位: {missing}")
        df = _coerce_types(df)
        # 基礎清理：去除非數值與非有限值
        for c in ["f_hz", "k", "delta_ct2", "sigma"]:
            df = df[df[c].apply(_is_finite)]
        frames.append(df[REQUIRED_COLS])

    cat = pd.concat(frames, axis=0, ignore_index=True)
    # 避免重複列（同事件/IFO/頻率）
    cat = cat.drop_duplicates(subset=["event", "ifo", "f_hz"], keep="first")
    # 排序（審稿友善）
    cat = cat.sort_values(by=["event", "ifo", "f_hz"]).reset_index(drop=True)
    cat.to_csv(out_csv, index=False)

    events = sorted(cat["event"].unique().tolist())
    summary = AggregateSummary(out_csv=out_csv,
                               n_files=len(fns),
                               n_rows=int(cat.shape[0]),
                               n_events=len(events),
                               events=events)
    if write_summary_json:
        with open(write_summary_json, "w") as f:
            json.dump(asdict(summary), f, indent=2)
    return summary
