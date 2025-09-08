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

    # ==== ここから追加（p_usd/p_btc の resample・ema 後）====

import numpy as np
import pandas as pd

SLOPE_H = 6  # 直近6時間の傾き。好みで 3/12 などに

def slope_over(series: pd.Series, hours: int) -> float:
    """直近 hours の平均傾き（1時間あたり）。データが薄い時も asof で補う。"""
    if series.empty:
        return np.nan
    t_now = series.index[-1]
    t_then = t_now - pd.Timedelta(hours=hours)
    v_then = series.asof(t_then)  # 直近 t_then 以下の値
    if pd.isna(v_then):
        v_then = series.iloc[0]
        dt_h = (series.index[-1] - series.index[0]).total_seconds()/3600 or 1.0
    else:
        dt_h = hours
    return (float(series.iloc[-1]) - float(v_then)) / dt_h

labels = []  # (sym, u_now, b_now, u_slope, b_slope, breakout_ts)

for sym in p_usd.columns:
    u = p_usd[sym].dropna()
    b = p_btc[sym].dropna()
    if u.empty or b.empty:
        continue

    u_now, b_now = float(u.iloc[-1]), float(b.iloc[-1])
    u_slope = slope_over(u, SLOPE_H)
    b_slope = slope_over(b, SLOPE_H)

    # ブレイクアウト（U>0 & B>0 が連続3ポイント以上になった最初の時刻）
    brk = None
    both_pos = (u > 0) & (b > 0)
    run = (both_pos != both_pos.shift()).cumsum()
    seglen = both_pos.groupby(run).transform('size')
    ok = both_pos & (seglen >= 3)
    if ok.any():
        brk = ok[ok].index[0]

    labels.append((sym, u_now, b_now, u_slope, b_slope, brk))

    # 並び順：いま強い(現在値)7割 + 勢い(傾き)3割
    labels.sort(key=lambda x: 0.7*(x[1]+x[2]) + 0.3*(x[3]+x[4]), reverse=True)
    
    def arrow(v):
        return "↗︎" if v > 0.02 else ("↘︎" if v < -0.02 else "→")
    
    x_end = p_usd.index[-1]
    pad = (p_usd.index[-1] - p_usd.index[0]) * 0.03
    
    for sym, u_now, b_now, us, bs, brk in labels:
        y = (u_now + b_now) / 2.0  # ラベルを置く高さ（USD/BTCの平均）
        txt = f"{sym}  U:{u_now:+.2f} / B:{b_now:+.2f}  {arrow(us)}{SLOPE_H}h:{us:+.2f}/{bs:+.2f}"
        ax.text(x_end + pad, y, txt, va="center", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"))
    
        if brk is not None:
            y_brk = (p_usd.loc[brk, sym] + p_btc.loc[brk, sym]) / 2.0
            ax.scatter([brk], [y_brk], s=28, zorder=5)
    
    # ==== 追加おわり ====


    plt.tight_layout()
    if not args.out:
        args.out = f"usd_btc_lines_h{args.hours}_top{len(keep)}.png"
    plt.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"[OK] wrote: {args.out}")

if __name__ == "__main__":
    main()
