#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Large-Cap Relative Strength (BTC-quoted)
# pip install requests pyyaml pandas matplotlib

import argparse
import requests
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import math
from pathlib import Path
import sys

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"

# ステーブル・WBTC などはデフォ除外
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

def fetch_top_mcap_ids(n=12, exclude=None):
    """時価総額上位から n 銘柄を返す（exclude を除外）"""
    if exclude is None:
        exclude = set()
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 250,
        "page": 1,
        "sparkline": "false",
    }
    r = requests.get(COINGECKO_URL, params=params, timeout=30)
    r.raise_for_status()
    rows = r.json()

    ids = []
    for x in rows:
        cid = x.get("id")
        if not cid:
            continue
        if cid in exclude:
            continue
        ids.append(cid)
        if len(ids) >= n:
            break
    return ids

def resolve_universe(cfg):
    """
    YAML の設定から最終的な id リストを作る。
    - manual: cfg['universe_ids'] を使用
    - top_mcap: CoinGecko から上位を自動取得（exclude/include を反映）
    """
    mode = (cfg.get("universe_mode") or "manual").lower()
    exclude = set(DEFAULT_EXCLUDE) | set(cfg.get("exclude_ids", []))
    include = list(cfg.get("include_ids", []))

    if mode == "top_mcap":
        n = int(cfg.get("top_mcap_n", 12))
        ids = fetch_top_mcap_ids(n=n, exclude=exclude)
        # include は先頭優先で重複排除
        ids = list(dict.fromkeys(include + ids))
        return ids
    else:
        ids = cfg.get("universe_ids")
        if not isinstance(ids, (list, tuple)) or not ids:
            raise ValueError("manual モードでは universe_ids を非空の配列で指定してください")
        # include を先頭に
        ids = list(dict.fromkeys(list(include) + list(ids)))
        # exclude は最後に適用（念のため）
        ids = [c for c in ids if c not in exclude]
        return ids

def zscore(series):
    s = pd.Series(series, dtype="float64")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or math.isnan(sd):
        return pd.Series([0]*len(s), index=s.index)
    return (s - mu) / sd

def compute_scores(df, weights, use_volume_factor, volume_factor_weight):
    z1  = zscore(df["pct_1h_btc"])
    z24 = zscore(df["pct_24h_btc"])
    z7  = zscore(df["pct_7d_btc"])
    comp = (
        weights.get("pct_1h_btc", 0.0)  * z1  +
        weights.get("pct_24h_btc", 0.0) * z24 +
        weights.get("pct_7d_btc", 0.0)  * z7
    )
    if use_volume_factor:
        import numpy as np
        zv = zscore(df["volume_usd"]).clip(-3, 3)
        vol_bonus = 1/(1 + np.exp(-zv)) - 0.5  # -0.5〜+0.5
        comp = comp + volume_factor_weight * vol_bonus
    return comp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_largecap_btc.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERR] config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # 最終的な universe を確定
    ids = resolve_universe(cfg)

    # 価格変化率は BTC 建て、出来高や USD 価格は USD から取得
    data_btc = fetch_markets(ids, vs="btc")
    data_usd = fetch_markets(ids, vs="usd")

    weights  = cfg.get("weights", {})
    use_vol  = bool(cfg.get("use_volume_factor", True))
    vol_w    = float(cfg.get("volume_factor_weight", 0.1))
    top_n    = int(cfg.get("top_n", 12))  # ← デフォルトを明示
    out_csv  = cfg.get("out_csv", "largecap_strength_btc.csv")
    out_png  = cfg.get("out_png", "largecap_strength_btc.png")

    # ---- DataFrame 構築 ----
    rows = []
    for x in data_btc:
        rows.append({
            "id": x.get("id"),
            "symbol": str(x.get("symbol", "")).upper(),
            "name": x.get("name"),
            "price_btc": x.get("current_price"),
            "pct_1h_btc":  (x.get("price_change_percentage_1h_in_currency")  or 0.0),
            "pct_24h_btc": (x.get("price_change_percentage_24h_in_currency") or 0.0),
            "pct_7d_btc":  (x.get("price_change_percentage_7d_in_currency")  or 0.0),
            "market_cap_rank": x.get("market_cap_rank"),
        })
    df = pd.DataFrame(rows)

    vol_map       = {x["id"]: x.get("total_volume")   for x in data_usd}
    mcap_map      = {x["id"]: x.get("market_cap")     for x in data_usd}
    price_usd_map = {x["id"]: x.get("current_price")  for x in data_usd}
    df["volume_usd"]     = df["id"].map(vol_map).fillna(0.0)
    df["market_cap_usd"] = df["id"].map(mcap_map).fillna(0.0)
    df["price_usd"]      = df["id"].map(price_usd_map).fillna(0.0)

    # ---- スコア計算 & 出力 ----
    df["score"] = compute_scores(df, weights, use_vol, vol_w)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    out_cols = [
        "rank","name","symbol","price_btc","price_usd",
        "pct_1h_btc","pct_24h_btc","pct_7d_btc",
        "market_cap_rank","market_cap_usd","volume_usd","score"
    ]
    df[out_cols].to_csv(out_csv, index=False)

    top = df.head(top_n)
    plt.figure(figsize=(10, 6))
    plt.bar(top["symbol"], top["score"])
    plt.title("Large-Cap RS (BTC-quoted)")
    plt.xlabel("Symbol")
    plt.ylabel("Score (z)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

    print(df[out_cols].head(top_n).to_string(index=False))
    print(f"\n[OK] wrote CSV: {out_csv}")
    print(f"[OK] wrote chart: {out_png}")

if __name__ == "__main__":
    main()
