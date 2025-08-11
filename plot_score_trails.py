#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 履歴CSVから usd_score / btc_score の推移をマルチラインで描画
# 使い方例:
#   python plot_score_trails.py --metric btc --hours 24 --topn 10 --smooth 3

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

def load_hist(path: Path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    # symbolは大文字に統一（CSVは大文字のはずだが念のため）
    df["symbol"] = df["symbol"].str.upper()
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="history_scores.csv")
    ap.add_argument("--metric", choices=["usd","btc"], default="btc",
                    help="描画するスコア（usd または btc）")
    ap.add_argument("--hours", type=int, default=24,
                    help="直近N時間に絞る（例：24）")
    ap.add_argument("--topn", type=int, default=12,
                    help="最終時点の値が高い順に上位N本を表示")
    ap.add_argument("--smooth", type=int, default=1,
                    help="移動平均の窓（1=なし）")
    args = ap.parse_args()

    path = (HERE/args.hist).resolve()
    if not path.exists():
        print(f"[ERR] 履歴が見つかりません: {path}  先に history_aggregate.py を実行してください。")
        return

    df = load_hist(path)

    # 時間で絞り込み
    if args.hours > 0:
        cutoff = df["timestamp"].max() - pd.Timedelta(hours=args.hours)
        df = df[df["timestamp"] >= cutoff].copy()

    # ピボット（行:timestamp, 列:symbol）
    col = "btc_score" if args.metric == "btc" else "usd_score"
    pv = df.pivot_table(index="timestamp", columns="symbol", values=col, aggfunc="last").sort_index()

    # 最終値の高い順で上位N本を残す
    last = pv.iloc[-1].dropna().sort_values(ascending=False)
    keep = list(last.head(args.topn).index)
    pv = pv[keep]

    # スムージング（移動平均）
    if args.smooth > 1:
        pv = pv.rolling(args.smooth, min_periods=1).mean()

    # 描画
    plt.figure(figsize=(11,6))
    for sym in pv.columns:
        plt.plot(pv.index, pv[sym], label=sym)
    plt.axhline(0, linewidth=1)  # ゼロライン
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.title(f"{args.metric.upper()}-score Trails (last {args.hours}h, top {args.topn})")
    plt.xlabel("Time"); plt.ylabel("Score (z)")
    plt.legend(ncol=6, fontsize=9)
    plt.tight_layout()
    out = HERE/f"score_trails_{args.metric}_h{args.hours}_top{args.topn}.png"
    plt.savefig(out, dpi=150)
    print(f"[OK] wrote: {out}")

if __name__ == "__main__":
    main()
