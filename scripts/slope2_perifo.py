# scripts/slope2_perifo.py
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from uclgw.eval.slopefit import do_fit, load_ct

def main():
    ap = argparse.ArgumentParser(description="Per-IFO / Per-Event slope-2 consistency")
    ap.add_argument("--data", default="data/ct/ct_bounds.csv")
    ap.add_argument("--profile", default="configs/profiles/lvk_o3.yaml")
    args = ap.parse_args()

    csv = Path(args.data)
    df = load_ct(csv)
    events = sorted(df["event"].unique())
    out = {"by_event": {}, "overall": {}}

    for ev in events:
        dfe = df[df["event"] == ev]
        ifos = sorted(dfe["ifo"].unique())
        ev_res = {"per_ifo": {}, "combined_wls": None}
        # per IFO
        for ifo in ifos:
            r = do_fit(csv, Path(args.profile), method="wls", ifo=ifo, event=ev)
            ev_res["per_ifo"][ifo] = r.__dict__

        # combined（以 WLS on all points）
        r_all = do_fit(csv, Path(args.profile), method="wls", event=ev)
        ev_res["combined_wls"] = r_all.__dict__
        out["by_event"][ev] = ev_res

    # overall: 若有多事件，可再合併
    out["overall"]["n_events"] = len(events)

    Path("reports").mkdir(exist_ok=True, parents=True)
    Path("reports/slope2_perifo.json").write_text(json.dumps(out, indent=2))
    print("Wrote reports/slope2_perifo.json")

if __name__ == "__main__":
    main()
