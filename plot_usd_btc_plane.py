#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# USD-score vs BTC-score の平面に「軌跡（trails）」を描く
#
# 例:
#   python plot_usd_btc_plane.py --hist data/trails_db.csv \
#     --hours 48 --resample 1H --ema 3 --topn 16 --highlight 6 \
#     --out data/largecap_usd_vs_btc_plane.png

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe

def load_hist(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ここでは“再正規化しない”。時刻は tz-aware で揃えるだけ
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp","symbol"])
    df["symbol"] = df["symbol"].str.upper()
    return df.sort_values(["timestamp","symbol"])

def make_xy_pivots(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p_usd = (df.pivot_table(index="timestamp", columns="symbol",
                            values="usd_score", aggfunc="last").sort_index())
    p_btc = (df.pivot_table(index="timestamp", columns="symbol",
                            values="btc_score", aggfunc="last").sort_index())
    return p_usd, p_btc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True)
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--resample", default="1H", help="例: 30min / 1H / 2H （空文字で無効）")
    ap.add_argument("--ema", type=int, default=3, help="EWM の span（0/1で無効）")
    ap.add_argument("--topn", type=int, default=16)
    ap.add_argument("--highlight", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = load_hist(Path(args.hist))
    if args.hours > 0:
        cutoff = df["timestamp"].max() - pd.Timedelta(hours=args.hours)
        df = df[df["timestamp"] >= cutoff].copy()
    if df.empty:
        print("[ERR] no data")
        return

    p_usd, p_btc = make_xy_pivots(df)

    # 時刻を union し、小さな穴だけ補間（fill_value は使わない）
    idx = p_usd.index.union(p_btc.index)
    U = p_usd.reindex(idx).interpolate(method="time", limit=2, limit_area="inside").ffill(limit=1).bfill(limit=1)
    B = p_btc.reindex(idx).interpolate(method="time", limit=2, limit_area="inside").ffill(limit=1).bfill(limit=1)

    # リサンプリング
    if args.resample:
        U = U.resample(args.resample).median()
        B = B.resample(args.resample).median()

    # 平滑化（必要なら）
    if args.ema and args.ema > 1:
        U = U.ewm(span=args.ema, adjust=False).mean()
        B = B.ewm(span=args.ema, adjust=False).mean()

    # 最新時点での上位銘柄を選ぶ（合成= usd+btc の絶対値で選抜）
    last_u = U.iloc[-1]; last_b = B.iloc[-1]
    score_for_pick = (last_u + last_b).abs()
    keep = list(score_for_pick.dropna().sort_values(ascending=False).head(args.topn).index)
    U, B = U[keep], B[keep]

    # 描画
    fig, ax = plt.subplots(figsize=(8, 8))
    # ガイド線
    ax.axvline(0, color="#2c6da4", lw=2)
    ax.axhline(0, color="#2c6da4", lw=2)

    # 強調とサブ
    k = max(1, min(args.highlight, len(keep)))
    strong, weak = keep[:k], keep[k:]

    # サブは薄く
    for sym in weak:
        x, y = U[sym].values, B[sym].values
        ax.plot(x, y, lw=1.0, alpha=0.25, color="0.6", zorder=1)

    # 強調は太く & ラベル
    for sym in strong:
        x, y = U[sym].values, B[sym].values
        ax.plot(x, y, lw=2.2, alpha=0.95, zorder=3, label=sym)
        # 終点ラベル
        ax.annotate(sym, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # 軸レンジはデータから自動（対称で少し余白）
    vmax = float(np.nanmax(np.abs(np.r_[U.to_numpy().ravel(), B.to_numpy().ravel()])))
    vmax = max(1.0, vmax) * 1.05
    ax.set_xlim(-vmax, vmax); ax.set_ylim(-vmax, vmax)

    ax.set_xlabel("USD composite (z)")
    ax.set_ylabel("BTC-quoted composite (z)")
    ax.set_title("USD-score vs BTC-score — trails")
    ax.grid(False)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[OK] wrote: {args.out}")

if __name__ == "__main__":
    main()

