#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Large-Cap Strength Screener
# USD建てのラージキャップ強弱スコアを算出
# pip install requests pyyaml pandas matplotlib

import argparse
import math
import sys
from pathlib import Path
import requests
import yaml
import pandas as pd
import matplotlib.pyplot as plt

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"

# デフォで外す（両YAMLに何も書かなくても安全に動く）
DEFAULT_EXCLUDE = {
    "tether","usd-coin","binance-usd","dai","first-digital-usd",
    "frax","usdd","terrausd","usde","paypal-usd","euro-coin",
    "wrapped-bitcoin","staked-ether"
}

def fetch_markets(ids, vs="usd"):
    params = {
        "vs_currency": vs,
        "ids": ",".join(ids),
        "sparkline": "false",
        "price_change_percentage": "1h,24h,7d",
    }
    r = requests.get(COINGECKO_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def zscore(series):
    s = pd.Series(series, dtype="float64")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or math.isnan(sd):
        return pd.Series([0]*len(s), index=s.index)
    return (s - mu) / sd

def compute_scores(df, weights, use_volume_factor, volume_factor_weight):
    z1 = zscore(df["pct_1h"])
    z24 = zscore(df["pct_24h"])
    z7 = zscore(df["pct_7d"])
    comp = (
        weights.get("pct_1h", 0.0) * z1 +
        weights.get("pct_24h", 0.0) * z24 +
        weights.get("pct_7d", 0.0) * z7
    )
    if use_volume_factor:
        import numpy as np
        zv = zscore(df["volume_usd"]).clip(-3, 3)
        vol_bonus = 1 / (1 + np.exp(-zv)) - 0.5
        comp = comp + volume_factor_weight * vol_bonus
    return comp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_largecap.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERR] config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ids = cfg["universe_ids"]
    weights = cfg.get("weights", {})
    use_volume_factor = bool(cfg.get("use_volume_factor", True))
    volume_factor_weight = float(cfg.get("volume_factor_weight", 0.1))
    top_n = int(cfg.get("top_n", 10))
    out_csv = cfg.get("out_csv", "largecap_strength.csv")
    out_png = cfg.get("out_png", "largecap_strength.png")

    data = fetch_markets(ids)
    if not data:
        print("[ERR] empty response from CoinGecko", file=sys.stderr)
        sys.exit(2)

    rows = []
    for x in data:
        rows.append({
            "id": x.get("id"),
            "symbol": str(x.get("symbol", "")).upper(),
            "name": x.get("name"),
            "price": x.get("current_price"),
            "pct_1h": (x.get("price_change_percentage_1h_in_currency") or 0.0),
            "pct_24h": (x.get("price_change_percentage_24h_in_currency") or 0.0),
            "pct_7d": (x.get("price_change_percentage_7d_in_currency") or 0.0),
            "market_cap": x.get("market_cap"),
            "market_cap_rank": x.get("market_cap_rank"),
            "volume_usd": x.get("total_volume"),
        })
    df = pd.DataFrame(rows)

    btc_row = df.loc[df["id"] == "bitcoin"]
    if not btc_row.empty:
        for col in ["pct_1h", "pct_24h", "pct_7d"]:
            df[f"{col}_vs_btc"] = df[col] - float(btc_row[col].values[0])
    else:
        df["pct_1h_vs_btc"] = 0.0
        df["pct_24h_vs_btc"] = 0.0
        df["pct_7d_vs_btc"] = 0.0

    df["score"] = compute_scores(df, weights, use_volume_factor, volume_factor_weight)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    out_cols = [
        "rank","name","symbol","price",
        "pct_1h","pct_24h","pct_7d",
        "pct_1h_vs_btc","pct_24h_vs_btc","pct_7d_vs_btc",
        "market_cap_rank","market_cap","volume_usd","score"
    ]
    df[out_cols].to_csv(out_csv, index=False)

    top = df.head(top_n)
    plt.figure(figsize=(10,6))
    plt.bar(top["symbol"], top["score"])
    plt.title("Large-Cap Relative Strength (Composite Score)")
    plt.xlabel("Symbol")
    plt.ylabel("Score (z)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

    print(df[out_cols].head(top_n).to_string(index=False))
    print(f"\\n[OK] wrote CSV: {out_csv}")
    print(f"[OK] wrote chart: {out_png}")

if __name__ == "__main__":
    main()
