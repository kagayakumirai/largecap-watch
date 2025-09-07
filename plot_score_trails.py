#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# score_trails（usd/btc）の可読性を高めた描画版
# 例:
#   python plot_score_trails.py --metric usd --hours 168 --topn 20 \
#     --highlight 8 --resample 30min --ema 5 --rank abs --right-labels --ylim 3.5

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

HERE = Path(__file__).parent

def load_hist(path: Path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["symbol"] = df["symbol"].str.upper()
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="history_scores.csv")
    ap.add_argument("--metric", choices=["usd", "btc", "usdxbtc"], default="btc",
                     help="描画するスコア（usd/btc/usdxbtc）")
    
    ap.add_argument("--hours", type=int, default=168, help="直近N時間に絞る")
    ap.add_argument("--topn", type=int, default=20, help="最終時点の値で上位N本を対象にする")
    # ★ 可読性向上オプション
    ap.add_argument("--rank", choices=["pos","abs"], default="pos",
                    help="上位抽出の基準: pos=値が大きい順, abs=絶対値が大きい順")
    ap.add_argument("--highlight", type=int, default=8,
                    help="上位K本を強調表示（残りはグレーで薄く）")
    ap.add_argument("--resample", default="30min",
                    help="時間リサンプリング（''で無効）例: 15min/30min/1H")
    ap.add_argument("--ema", type=int, default=5,
                    help="EMAスパン（0/1で無効）")
    ap.add_argument("--right-labels", action="store_true",
                    help="右端に銘柄名と値を注記（凡例の代替）")
    ap.add_argument("--ylim", type=float, default=3.5, help="y軸の±上限（クリップ）")
    ap.add_argument("--smooth", type=int, default=1, help="従来の移動平均（互換用・通常は0/1で）")
    args = ap.parse_args()

    path = (HERE/args.hist).resolve()
    if not path.exists():
        print(f"[ERR] 履歴が見つかりません: {path}")
        return

    df = load_hist(path)

    # 期間で絞り込み
    if args.hours > 0:
        cutoff = df["timestamp"].max() - pd.Timedelta(hours=args.hours)
        df = df[df["timestamp"] >= cutoff].copy()

    if args.metric in ("usd","btc"):
        col = "btc_score" if args.metric == "btc" else "usd_score"
        pv = (df.pivot_table(index="timestamp", columns="symbol", values=col, aggfunc="last")
                .sort_index())
    else:
        # usdxbtc = usd_score - btc_score（同じタイムスタンプでの差）
        p_usd = df.pivot_table(index="timestamp", columns="symbol", values="usd_score", aggfunc="last")
        p_btc = df.pivot_table(index="timestamp", columns="symbol", values="btc_score", aggfunc="last")
        pv = (p_usd - p_btc).sort_index()


    # リサンプリング（中央値）
    if args.resample:
        # DatetimeIndex 前提。念のため try にして落ちないように。
        try:
            pv = pv.resample(args.resample).median()
        except Exception:
            pass

    # EMAスムージング（軽くノイズ除去）
    if args.ema and args.ema > 1:
        pv = pv.ewm(span=args.ema, adjust=False).mean()

    # （互換）単純移動平均
    if args.smooth and args.smooth > 1:
        pv = pv.rolling(args.smooth, min_periods=1).mean()

    # 最終時点の上位Nを抽出（pos or abs）
    last = pv.iloc[-1].dropna()
    if args.rank == "abs":
        last = last.reindex(last.abs().sort_values(ascending=False).index)
    else:
        last = last.sort_values(ascending=False)
    keep = list(last.head(args.topn).index)
    pv = pv[keep]

    # 強調対象
    k = max(1, min(args.highlight, len(keep)))
    topK = keep[:k]
    others = keep[k:]

    # 描画
    plt.figure(figsize=(12, 4.5))
    ax = plt.gca()

    # ガイドライン ±2, ±1, 0
    for y in (-2, -1, 0, 1, 2):
        ax.axhline(y, lw=0.8 if y==0 else 0.6,
                   ls="--", alpha=0.45 if y else 0.7, zorder=1)

    # others: グレーで薄く
    for sym in others:
        ax.plot(pv.index, pv[sym], lw=0.8, alpha=0.25, color="0.5", zorder=2)

    # topK: 太め＆はっきり
    for sym in topK:
        ax.plot(pv.index, pv[sym], lw=1.8, alpha=0.95, zorder=3, label=sym)

    # 右端ラベル（凡例の代わり）
    if args.right_labels:
        x_last = pv.index[-1]
        for i, sym in enumerate(topK):
            yv = pv[sym].iloc[-1]
            # 軽い衝突回避
            yv_shift = yv + (0.06 * (len(topK)//2 - i))
            txt = f"{sym}  {yv:+.2f}"
            ax.annotate(txt, xy=(x_last, yv), xytext=(6, 0),
                        textcoords="offset points", va="center",
                        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                        fontsize=9)

    ax.set_ylim(-args.ylim, args.ylim)
    ax.set_xlabel("Time"); ax.set_ylabel("Score (z)")
    ax.set_title(f"{args.metric.upper()}-score Trails (last {args.hours}h, top {args.topn})")

    # 目盛り調整
    if args.hours >= 72:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")

    # 凡例は必要なら表示（右端ラベル使用時はOFF推奨）
    if not args.right_labels:
        ax.legend(ncol=6, fontsize=9, frameon=False)

    plt.tight_layout()
    out = HERE/f"score_trails_{args.metric}_h{args.hours}_top{args.topn}_clean.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[OK] wrote: {out}")

if __name__ == "__main__":
    main()
