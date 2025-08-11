#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# フォルダ内の largecap_compare_YYYYMMDD_HHMM.csv を集めて1つの履歴に統合

from pathlib import Path
import re
import pandas as pd

HERE = Path(__file__).parent
PAT = re.compile(r"largecap_compare_(\d{8})_(\d{4})\.csv")

def parse_stamp(name: str):
    m = PAT.fullmatch(name)
    if not m: return None
    d, t = m.groups()
    ts = pd.to_datetime(f"{d} {t}", format="%Y%m%d %H%M")
    return ts

def main():
    files = sorted((HERE.glob("largecap_compare_*.csv")), key=lambda p: p.name)
    rows = []
    for f in files:
        ts = parse_stamp(f.name)
        if ts is None: continue
        df = pd.read_csv(f)
        df["timestamp"] = ts
        rows.append(df)
    if not rows:
        print("[WARN] 履歴CSVが見つかりません。まず scheduler でスナップショットを溜めてください。")
        return
    hist = pd.concat(rows, ignore_index=True)
    out_csv = HERE/"history_scores.csv"
    out_parq = HERE/"history_scores.parquet"
    hist.to_csv(out_csv, index=False)
    try:
        hist.to_parquet(out_parq, index=False)
    except Exception:
        pass
    print(f"[OK] wrote: {out_csv}  (rows={len(hist)})")

if __name__ == "__main__":
    main()
