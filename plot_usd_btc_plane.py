#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# USD-score vs BTC-score の「時間軌跡（折れ線）」を描く
# 例:
#   python plot_usd_btc_plane.py --hist data/trails_db.csv \
#     --hours 48 --resample 1H --ema 3 --topn 16 --highlight 6 \
#     --out data/largecap_usd_vs_btc.png

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_hist(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="data/trails_db.csv")
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--resample", default="1H", help="例: 15min/30min/1H（''で無効）")
    ap.add_argument("--ema", type=int, default=3, help="EMAスパン（0/1で無効）")
    ap.add_argument("--topn", type=int, default=16, help="最終時点で対象にする銘柄数")
    ap.add_argument("--highlight", type=int, default=6, help="太線にする本数")
    ap.add_argument("--lam", type=float, default=0.3, help="不均衡ペナルティ λ（選別用）")
    ap.add_argument("--out", default="largecap_usd_vs_btc_lines.png")
    args = ap.parse_args()

    path = Path(args.hist).resolve()
    if not path.exists():
        print(f"[ERR] not found: {path}")
        return
    df = load_hist(path)

    # 時間範囲
    if args.hours > 0:
        cutoff = df["timestamp"].max() - pd.Timedelta(hours=args.hours)
        df = df[df["timestamp"] >= cutoff].copy()
    if df.empty:
        print("[ERR] empty after time filter"); return

    # ピボット（USD / BTC）
    p_usd = (df.pivot_table(index="timestamp", columns="symbol", values="usd_score", aggfunc="last")
               .sort_index())
    p_btc = (df.pivot_table(index="timestamp", columns="symbol", values="btc_score", aggfunc="last")
               .sort_index())

    # リサンプリング（同一ルールで揃うので index は一致）
    if args.resample:
        p_usd = p_usd.resample(args.resample).median()
        p_btc = p_btc.resample(args.resample).median()

    # 小さな穴の補完 + EMA
    def _smooth(pv: pd.DataFrame) -> pd.DataFrame:
        pv = pv.interpolate(method="time", limit=4, limit_area="inside").ffill(limit=2).bfill(limit=2)
        if args.ema and args.ema > 1:
            pv = pv.ewm(span=args.ema, adjust=False).mean()
        return pv
    p_usd, p_btc = _smooth(p_usd), _smooth(p_btc)

    # 最終点での並べ替え（sum - lam*|diff| を採用）
    last_usd = p_usd.iloc[-1]
    last_btc = p_btc.iloc[-1]
    score = (last_usd + last_btc) - args.lam * (last_usd - last_btc).abs()
    keep = list(score.dropna().sort_values(ascending=False).head(args.topn).index)

    # 描画
    fig, ax = plt.subplots(figsize=(8, 6))

    # others（薄く）
    k = max(1, min(args.highlight, len(keep)))
    topK, others = keep[:k], keep[k:]

    def _plot_one(sym: str, lw: float, alpha: float, z: int):
        x = p_usd[sym]
        y = p_btc[sym]
        xy = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        if len(xy) >= 2:
            ax.plot(xy["x"], xy["y"], lw=lw, alpha=alpha, zorder=z, label=sym)

    for sym in others:
        _plot_one(sym, lw=0.8, alpha=0.25, z=2)
    for sym in topK:
        _plot_one(sym, lw=1.8, alpha=0.95, z=3)

    # 最終点にラベル
    for sym in topK:
        if sym in p_usd.columns and sym in p_btc.columns:
            xv = p_usd[sym].iloc[-1]
            yv = p_btc[sym].iloc[-1]
            if pd.notna(xv) and pd.notna(yv):
                ax.plot([xv], [yv], marker="o", ms=3, zorder=4)
                ax.text(xv, yv, f" {sym}", va="center", fontsize=8)

    # 軸・ガイド
    ax.axhline(0, lw=1)
    ax.axvline(0, lw=1)
    ax.set_xlabel("USD composite (z)")
    ax.set_ylabel("BTC-quoted composite (z)")
    ax.set_title("USD-score vs BTC-score — trails")

    # レンジ（対称に）
    allx = np.concatenate([p_usd[c].to_numpy(dtype=float) for c in keep]) if keep else np.array([0.0])
    ally = np.concatenate([p_btc[c].to_numpy(dtype=float) for c in keep]) if keep else np.array([0.0])
    xmax = np.nanmax(np.abs(allx)); ymax = np.nanmax(np.abs(ally))
    m = float(max(1e-6, xmax, ymax)) * 1.05
    ax.set_xlim(-m, m); ax.set_ylim(-m, m)

    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[OK] wrote: {out}")

if __name__ == "__main__":
    main()
