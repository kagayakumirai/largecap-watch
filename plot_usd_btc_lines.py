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
    return df.sort_values(["timestamp", "symbol"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="data/trails_db.csv")
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--resample", default="1H")
    ap.add_argument("--ema", type=int, default=3)
    ap.add_argument("--topn", type=int, default=16)
    ap.add_argument("--rank", choices=["pos", "abs"], default="abs")
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
    p_usd = df.pivot_table(index="timestamp", columns="symbol",
                           values="usd_score", aggfunc="last").sort_index()
    p_btc = df.pivot_table(index="timestamp", columns="symbol",
                           values="btc_score", aggfunc="last").sort_index()

    # リサンプリング（中央値）→ 小穴補間 → 軽くスムージング
    def prep(pv: pd.DataFrame) -> pd.DataFrame:
        if args.resample:
            pv = pv.resample(args.resample).median()
        pv = (
            pv.interpolate(method="time", limit=4, limit_area="inside")
              .ffill(limit=2)
              .bfill(limit=2)
        )
        if args.ema and args.ema > 1:
            pv = pv.ewm(span=args.ema, adjust=False).mean()
        return pv

    p_usd, p_btc = prep(p_usd), prep(p_btc)

    # 上位選定：直近の USD/BTC を合成（abs で強い順 or 値で強い順）
    last_usd = p_usd.iloc[-1]
    last_btc = p_btc.iloc[-1]
    combo = (last_usd.abs() + last_btc.abs()) if args.rank == "abs" else (last_usd + last_btc)
    keep = list(combo.dropna().sort_values(ascending=False).head(args.topn).index)
    p_usd, p_btc = p_usd[keep], p_btc[keep]

    # 描画
    fig, ax = plt.subplots(figsize=(args.figw, args.figh))

    # ガイドライン
    for y in (-2, -1, 0, 1, 2):
        ax.axhline(y, lw=0.8 if y == 0 else 0.6, ls="--",
                   alpha=0.45 if y else 0.7, zorder=1)

    # 色サイクルから銘柄ごとの色を決めて保存（USD 実線 / BTC 破線を同色）
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_by_sym = {}

    for i, sym in enumerate(keep):
        c = colors[i % len(colors)]
        color_by_sym[sym] = c
        ax.plot(p_usd.index, p_usd[sym].values, lw=1.8, color=c, zorder=3)
        ax.plot(p_btc.index, p_btc[sym].values, lw=1.2, ls="--", alpha=0.95,
                color=c, zorder=3)
    # ★ここから追加：右側に余白（ラベル用のパッド）を確保
    span = p_usd.index[-1] - p_usd.index[0]
    pad  = span * 0.08   # ← もともと 0.03 くらい。右へ大きく寄せたいので 0.08
    ax.set_xlim(p_usd.index[0], p_usd.index[-1] + pad*1.25)

    # yレンジ（ラベル配置で使うのでここで確定）
    if args.ylim is not None:
        ax.set_ylim(-args.ylim, args.ylim)
    else:
        vmax = np.nanmax(np.abs(np.concatenate([p_usd.values.ravel(), p_btc.values.ravel()])))
        vmax = 1.05 * (vmax if np.isfinite(vmax) else 1.0)
        ax.set_ylim(-vmax, vmax)

    ax.set_xlabel("Time")
    ax.set_ylabel("Score (z)")
    ax.set_title(f"USD & BTC composites (last {args.hours}h, top {len(keep)})")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    # 凡例は混みやすいのでデフォルトOFF（右端ラベル推奨）
    # ax.legend(ncol=4, fontsize=8, frameon=False)

    # ===== 右端ラベル（色付き■＋重なり回避） =====
    SLOPE_H = 6  # 直近6時間の傾き

    def value_at(series: pd.Series, ts) -> float:
        """時間 ts の値（なければ time 補間で推定）"""
        s = series.copy()
        if ts not in s.index:
            s.loc[ts] = np.nan
            s = s.sort_index().interpolate(method="time")
        return float(s.loc[ts])

    def slope_over(series: pd.Series, hours: int) -> float:
        if series.empty:
            return np.nan
        t_now = series.index[-1]
        t_then = t_now - pd.Timedelta(hours=hours)
        v_then = value_at(series, t_then)
        dt_h = (t_now - t_then).total_seconds() / 3600.0
        if dt_h <= 0:
            dt_h = 1.0
        return (float(series.iloc[-1]) - float(v_then)) / dt_h

    def arrow(v: float) -> str:
        return "↗︎" if v > 0.02 else ("↘︎" if v < -0.02 else "→")

    labels = []
    for sym in p_usd.columns:
        u = p_usd[sym].dropna()
        b = p_btc[sym].dropna()
        if u.empty or b.empty:
            continue
        u_now, b_now = float(u.iloc[-1]), float(b.iloc[-1])
        u_slope, b_slope = slope_over(u, SLOPE_H), slope_over(b, SLOPE_H)

        # ブレイクアウト（U>0 & B>0 が連続3ポイント以上になった最初の時刻）
        brk = None
        both_pos = (u > 0) & (b.reindex(u.index).ffill() > 0)
        run = (both_pos != both_pos.shift()).cumsum()
        seglen = both_pos.groupby(run).transform("size")
        ok = both_pos & (seglen >= 3)
        if ok.any():
            brk = ok[ok].index[0]

        score_for_order = 0.7*(u_now+b_now) + 0.3*(u_slope+b_slope)
        labels.append((score_for_order, sym, u_now, b_now, u_slope, b_slope, brk))

    labels.sort(reverse=True)

    def spread_labels(yvals, min_gap):
        if not yvals:
            return yvals
        order = np.argsort(yvals)
        placed = {}
        prev = None
        for idx in order:
            y = yvals[idx]
            val = y if prev is None else max(y, prev + min_gap)
            placed[idx] = val
            prev = val
        return [placed[i] for i in range(len(yvals))]

    if args.right_labels and labels:
        x_end = p_usd.index[-1]
        pad = (x_end - p_usd.index[0]) * 0.03
        y_anchor = [ (u_now + b_now)/2.0 for _,_,u_now,b_now,_,_,_ in labels ]
        yrng = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_spread = spread_labels(y_anchor, min_gap=0.06*yrng)  # 行間は好みで

        for y_adj, (_, sym, u_now, b_now, us, bs, brk) in zip(y_spread, labels):
            c = color_by_sym.get(sym, "0.2")
            # 色付き■
            ax.text(x_end + pad, y_adj, u"\u25A0", color=c, va="center",
                    fontsize=10, clip_on=False, zorder=6)
            # ラベル本文
            txt = f"  {sym}  U:{u_now:+.2f} / B:{b_now:+.2f}  {arrow(us)}{SLOPE_H}h:{us:+.2f}/{bs:+.2f}"
            ax.text(x_end + pad, y_adj, txt, va="center", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=0.4),
                    clip_on=False, zorder=5)
            # ブレイクアウト印
            if brk is not None:
                y_brk = float(np.nanmean([value_at(p_usd[sym], brk), value_at(p_btc[sym], brk)]))
                ax.scatter([brk], [y_brk], s=28, color=c, zorder=5)

    # 保存
    plt.tight_layout()
    if not args.out:
        args.out = f"usd_btc_lines_h{args.hours}_top{len(keep)}.png"
    plt.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"[OK] wrote: {args.out}")

if __name__ == "__main__":
    main()
