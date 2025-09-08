#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# USD/BTC の合成スコアを時間軸で並べた折れ線図
# 例:
#   python plot_usd_btc_lines.py --hist data/trails_db.csv \
#     --hours 48 --resample 1H --ema 3 --topn 16 --rank abs --right-labels

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe

HERE = Path(__file__).parent

def load_hist(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df = df.sort_values(["timestamp","symbol"])
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="data/trails_db.csv")
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--resample", default="1H")
    ap.add_argument("--ema", type=int, default=3)
    ap.add_argument("--topn", type=int, default=16)
    ap.add_argument("--rank", choices=["pos","abs"], default="abs")
    ap.add_argument("--right-labels", action="store_true")
    ap.add_argument("--figw", type=float, default=12)
    ap.add_argument("--figh", type=float, default=4.8)
    ap.add_argument("--ylim", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = Path(args.hist)
    if not p.exists():
        print(f"[ERR] not found: {p}")
        return
    df = load_hist(p)

    # 時間窓
    if args.hours > 0:
        cutoff = df["timestamp"].max() - pd.Timedelta(hours=args.hours)
        df = df[df["timestamp"] >= cutoff].copy()

    # ピボット（USD / BTC）
    p_usd = df.pivot_table(index="timestamp", columns="symbol", values="usd_score", aggfunc="last").sort_index()
    p_btc = df.pivot_table(index="timestamp", columns="symbol", values="btc_score", aggfunc="last").sort_index()

    # リサンプリング（中央値）→ 小穴補間 → 軽くスムージング
    def prep(pv):
        if args.resample: pv = pv.resample(args.resample).median()
        pv = pv.interpolate(method="time", limit=4, limit_area="inside").ffill(limit=2).bfill(limit=2)
        if args.ema and args.ema > 1: pv = pv.ewm(span=args.ema, adjust=False).mean()
        return pv
    p_usd, p_btc = prep(p_usd), prep(p_btc)

    # 上位選定：直近の USD/BTC を合成（abs で強い順 or 値で強い順）
    last_usd = p_usd.iloc[-1]
    last_btc = p_btc.iloc[-1]
    combo = (last_usd.abs() + last_btc.abs()) if args.rank=="abs" else (last_usd + last_btc)
    keep = list(combo.dropna().sort_values(ascending=False).head(args.topn).index)

    p_usd, p_btc = p_usd[keep], p_btc[keep]

    # 描画
    fig, ax = plt.subplots(figsize=(args.figw, args.figh))

    # ガイドライン
    for y in (-2,-1,0,1,2):
        ax.axhline(y, lw=0.8 if y==0 else 0.6, ls="--", alpha=0.45 if y else 0.7, zorder=1)

    # 色は Matplotlib デフォルト循環を使用
    for i, sym in enumerate(keep):
        y_u = p_usd[sym].values
        y_b = p_btc[sym].values
        ax.plot(p_usd.index, y_u, lw=1.8, label=f"{sym} USD", zorder=3)
        ax.plot(p_btc.index, y_b, lw=1.2, ls="--", alpha=0.95, label=f"{sym} BTC", zorder=3)

    # 右端ラベル
    if args.right_labels:
        x_last = p_usd.index[-1]
        for sym in keep:
            u = float(p_usd[sym].iloc[-1]) if pd.notna(p_usd[sym].iloc[-1]) else np.nan
            b = float(p_btc[sym].iloc[-1]) if pd.notna(p_btc[sym].iloc[-1]) else np.nan
            if np.isfinite(u) or np.isfinite(b):
                y_anchor = np.nanmean([u,b])
                txt = f"{sym}  U:{u:+.2f} / B:{b:+.2f}"
                ax.annotate(txt, xy=(x_last, y_anchor), xytext=(6,0), textcoords="offset points",
                            va="center", fontsize=9,
                            path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # yレンジ
    if args.ylim is not None:
        ax.set_ylim(-args.ylim, args.ylim)
    else:
        vmax = np.nanmax(np.abs(np.concatenate([p_usd.values.ravel(), p_btc.values.ravel()])))
        vmax = 1.05 * (vmax if np.isfinite(vmax) else 1.0)
        ax.set_ylim(-vmax, vmax)

    ax.set_xlabel("Time"); ax.set_ylabel("Score (z)")
    ax.set_title(f"USD & BTC composites (last {args.hours}h, top {len(keep)})")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")

    # 凡例は混みやすいのでデフォルトOFF（右端ラベル推奨）
    # ax.legend(ncol=4, fontsize=8, frameon=False)

    plt.tight_layout()
    if not args.out:
        args.out = f"usd_btc_lines_h{args.hours}_top{len(keep)}.png"
    plt.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"[OK] wrote: {args.out}")

if __name__ == "__main__":
    main()
