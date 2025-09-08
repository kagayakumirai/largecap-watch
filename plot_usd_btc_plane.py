#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# USD-score vs BTC-score の「折れ線トレイル」版

import argparse
from pathlib import Path
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _last_per_symbol(df: pd.DataFrame) -> pd.DataFrame:
    # 各 symbol の“最後の観測行”を取得
    return df.sort_values("timestamp").groupby("symbol", as_index=False).tail(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="data/trails_db.csv")
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--resample", default="1H", help="例: 15min/30min/1H/6H/1D")
    ap.add_argument("--ema", type=int, default=0, help="0/1で無効、2以上で平滑")
    ap.add_argument("--topn", type=int, default=16)
    ap.add_argument("--highlight", type=int, default=6)
    ap.add_argument("--rank", choices=["mag","sum","usd","btc"], default="mag",
                    help="上位抽出の指標: mag=√(usd^2+btc^2), sum=|usd|+|btc|")
    ap.add_argument("--out", default="data/largecap_usd_vs_btc_trails.png")
    args = ap.parse_args()

    p = Path(args.hist)
    if not p.exists():
        print(f"[ERR] not found: {p}")
        return

    df = pd.read_csv(p, parse_dates=["timestamp"])
    # 時間フィルタ
    if args.hours > 0 and not df.empty:
        cutoff = df["timestamp"].max() - pd.Timedelta(hours=args.hours)
        df = df[df["timestamp"] >= cutoff].copy()

    if df.empty:
        print("[ERR] empty history after filtering")
        return

    # 直近での順位付けに使う値
    last = _last_per_symbol(df)
    if args.rank == "mag":
        score = np.sqrt(last["usd_score"]**2 + last["btc_score"]**2)
    elif args.rank == "sum":
        score = np.abs(last["usd_score"]) + np.abs(last["btc_score"])
    elif args.rank == "usd":
        score = np.abs(last["usd_score"])
    else:
        score = np.abs(last["btc_score"])
    keep = last.assign(_rank=score).sort_values("_rank", ascending=False)["symbol"].head(args.topn).tolist()

    df = df[df["symbol"].isin(keep)].copy()

    # 図
    fig, ax = plt.subplots(figsize=(7.5, 8.0))

    # 目安線
    ax.axhline(0, color="#2b5a7b", lw=2)
    ax.axvline(0, color="#2b5a7b", lw=2)

    # ハイライト対象
    k = max(1, min(args.highlight, len(keep)))
    hi = set(keep[:k])

    # 銘柄ごとに resample → ema 平滑 → 折れ線
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("timestamp").set_index("timestamp")[["usd_score","btc_score"]]
        if args.resample:
            g = g.resample(args.resample).median()
        if args.ema and args.ema > 1:
            g = g.ewm(span=args.ema, adjust=False).mean()
        g = g.dropna(how="any")
        if g.empty:
            continue

        lw = 2.2 if sym in hi else 1.0
        alpha = 0.95 if sym in hi else 0.25
        ax.plot(g["usd_score"], g["btc_score"], lw=lw, alpha=alpha, zorder=2 if sym in hi else 1)

        # 右端（最新点）にシンボルラベル
        x, y = float(g["usd_score"].iloc[-1]), float(g["btc_score"].iloc[-1])
        if sym in hi:
            ax.text(x, y, sym, fontsize=9, ha="left", va="center", xytext=(4,0),
                    textcoords="offset points")

    ax.set_xlabel("USD composite (z)")
    ax.set_ylabel("BTC-quoted composite (z)")
    ax.set_title("USD-score vs BTC-score — trails")

    # 対称レンジに（少し余白）
    allx = df["usd_score"].to_numpy(np.float64)
    ally = df["btc_score"].to_numpy(np.float64)
    vmax = max(np.nanmax(np.abs(allx)), np.nanmax(np.abs(ally)))
    vmax = 1.05 * (vmax if np.isfinite(vmax) else 1.0)
    ax.set_xlim(-vmax, vmax)
    ax.set_ylim(-vmax, vmax)

    ax.grid(False)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"[OK] wrote: {args.out}")

if __name__ == "__main__":
    main()
