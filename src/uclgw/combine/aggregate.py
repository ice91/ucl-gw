# src/uclgw/combine/aggregate.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

COLS = ["event","ifo","f_hz","k","delta_ct2","sigma"]

def append_event_to_master(event_csv: Path, master_csv: Path) -> None:
    event_csv = Path(event_csv)
    master_csv = Path(master_csv)
    df = pd.read_csv(event_csv)
    df = df[COLS].copy()

    if master_csv.exists():
        dm = pd.read_csv(master_csv)
        cat = pd.concat([dm[COLS], df], ignore_index=True)
        cat = cat.drop_duplicates(subset=["event","ifo","f_hz"]).sort_values(["event","ifo","f_hz"])
        cat.to_csv(master_csv, index=False)
    else:
        df.to_csv(master_csv, index=False)
