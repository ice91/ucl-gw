# src/uclgw/combine/aggregate.py
from __future__ import annotations
import os, glob, math, json
from dataclasses import dataclass, asdict
from typing import List
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

def load_event_csvs(events_dir: str | os.PathLike) -> List[str]:
    events_dir = os.fspath(events_dir)
    return sorted(glob.glob(os.path.join(events_dir, "*_ct_bounds.csv")))

def aggregate(events_dir: str | os.PathLike = "data/ct/events",
              out_csv: str | os.PathLike = "data/ct/ct_bounds.csv",
              write_summary_json: str | os.PathLike | None = None) -> AggregateSummary:
    # 統一將 PathLike 轉為字串，避免 JSON dump 出錯
    events_dir = os.fspath(events_dir)
    out_csv = os.fspath(out_csv)
    write_summary_json = os.fspath(write_summary_json) if write_summary_json is not None else None

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fns = load_event_csvs(events_dir)
    if not fns:
        # 建立空框架以利 downstream 腳本處理
        pd.DataFrame(columns=REQUIRED_COLS).to_csv(out_csv, index=False)
        summary = AggregateSummary(out_csv=str(out_csv), n_files=0, n_rows=0, n_events=0, events=[])
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
    summary = AggregateSummary(out_csv=str(out_csv),
                               n_files=len(fns),
                               n_rows=int(cat.shape[0]),
                               n_events=len(events),
                               events=events)
    if write_summary_json:
        with open(write_summary_json, "w") as f:
            json.dump(asdict(summary), f, indent=2)
    return summary

def append_event_points(event_csv: str | os.PathLike,
                        out_csv: str | os.PathLike = "data/ct/ct_bounds.csv",
                        report_path: str | os.PathLike | None = None) -> AggregateSummary:
    """
    將單一事件 CSV 併入彙總 out_csv。
    內部直接呼叫 aggregate() 掃描 event_csv 所在資料夾，避免重複與型別不一致問題。
    """
    event_csv = os.fspath(event_csv)
    out_csv = os.fspath(out_csv)
    report_path = os.fspath(report_path) if report_path is not None else None

    events_dir = os.path.dirname(os.path.abspath(event_csv))
    return aggregate(events_dir=events_dir,
                     out_csv=out_csv,
                     write_summary_json=report_path)
