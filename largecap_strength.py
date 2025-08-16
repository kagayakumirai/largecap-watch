#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Large-Cap Strength Screener (USD)
# - CoinGeckoからラージキャップを取得し、合成スコアでランキング
# - 「上位N銘柄を自動選定(top_mcap)」/「手動リスト(manual)」を選べます
# pip install requests pyyaml pandas matplotlib

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import yaml
import os, time, requests

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"

# 何も書かなくても安全に動く“デフォ除外”
DEFAULT_EXCLUDE = {
    # 主要ステーブル
    "tether", "usd-coin", "binance-usd", "dai", "first-digital-usd",
    "frax", "usdd", "terrausd", "usde", "paypal-usd", "euro-coin",
    # ペグ/ラップ/ステーキング代表
    "wrapped-bitcoin", "staked-ether",
}

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
UNIVERSE_CACHE = DATA_DIR / "universe_cache.json"


# ------------------------- helpers -------------------------
def _cg_get(params, tries=5):
    # APIキーをSecretsから使う場合（任意）
    headers = {}
    api_key = os.getenv("COINGECKO_API_KEY") or params.pop("x_cg_api_key", None)
    if api_key:
        # プランによりヘッダ名が異なるため両方入れておく（どちらかが有効）
        headers["x-cg-pro-api-key"] = api_key
        headers["x-cg-demo-api-key"] = api_key

    for i in range(tries):
        r = requests.get(COINGECKO_URL, params=params, headers=headers, timeout=30)
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()
        wait = int(r.headers.get("Retry-After", "0")) or min(60, 2 ** i * 2)
        print(f"[WARN] 429 from CoinGecko. wait {wait}s and retry {i+1}/{tries}", flush=True)
        time.sleep(wait)
    raise RuntimeError("CoinGecko 429: exceeded retries")

def fetch_markets(ids, vs="usd"):
    params = {
        "vs_currency": vs,
        "ids": ",".join(ids),
        "sparkline": "false",
        "price_change_percentage": "1h,24h,7d",
    }
    return _cg_get(params)


def fetch_top_mcap_ids(n, exclude, retries=3, backoff=3):
    """時価総額降順の上位 n 個の id を返す（exclude は id の set）。失敗時はキャッシュにフォールバック。"""
    params = dict(
        vs_currency="usd",
        order="market_cap_desc",
        per_page=250,
        page=1,
        price_change_percentage="1h,24h,7d",
    )
    last_err = None
    for k in range(retries):
        try:
            r = requests.get(COINGECKO_URL, params=params, timeout=30)
            # レート制限
            if r.status_code == 429:
                time.sleep(backoff * (k + 1))
                continue
            r.raise_for_status()
            rows = r.json()
            ids = []
            for c in rows:
                cid = c.get("id")
                if not cid or cid in exclude:
                    continue
                ids.append(cid)
                if len(ids) >= n:
                    break
            if not ids:
                raise RuntimeError("empty ids from coingecko")
            UNIVERSE_CACHE.write_text(
                json.dumps({"ts": time.time(), "ids": ids}, ensure_ascii=False),
                encoding="utf-8",
            )
            return ids
        except Exception as e:
            last_err = e
            time.sleep(backoff * (k + 1))
    # フォールバック（前回の値を使う）
    if UNIVERSE_CACHE.exists():
        cached = json.loads(UNIVERSE_CACHE.read_text(encoding="utf-8"))
        ids = cached.get("ids") or []
        if ids:
            print("[WARN] Using cached universe:", ids)
            return ids[:n]
    raise RuntimeError(f"fetch_top_mcap_ids failed: {last_err}")


def resolve_universe(cfg: dict):
    """
    YAMLの設定からユニバース（取得対象の id リスト）を確定する。
      universe_mode: "top_mcap" or "manual"
      top_mcap_n: int
      exclude_ids: [id, ...]  # 追加除外
      include_ids: [id, ...]  # 必ず含める
      universe_ids: [id, ...] # manual のときだけ使用
    """
    mode = str(cfg.get("universe_mode", "manual")).lower()
    if mode == "top_mcap":
        n = int(cfg.get("top_mcap_n", 20))
        exclude = set(DEFAULT_EXCLUDE) | set(cfg.get("exclude_ids", []))
        include = list(cfg.get("include_ids", []))

        ids = fetch_top_mcap_ids(n, exclude)
        # include は重複しないように後ろに追加
        for x in include:
            if x not in ids:
                ids.append(x)
        print(f"[INFO] universe_mode=top_mcap  selected={len(ids)}  ids={ids}")
        return ids
    else:
        ids = list(cfg.get("universe_ids", []))
        print(f"[INFO] universe_mode=manual  selected={len(ids)}  ids={ids}")
        return ids


def zscore(series):
    s = pd.Series(series, dtype="float64")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or math.isnan(sd):
        return pd.Series([0] * len(s), index=s.index)
    return (s - mu) / sd


def compute_scores(df, weights, use_volume_factor, volume_factor_weight):
    z1 = zscore(df["pct_1h"])
    z24 = zscore(df["pct_24h"])
    z7 = zscore(df["pct_7d"])
    comp = (
        weights.get("pct_1h", 0.0) * z1
        + weights.get("pct_24h", 0.0) * z24
        + weights.get("pct_7d", 0.0) * z7
    )
    if use_volume_factor:
        import numpy as np

        zv = zscore(df["volume_usd"]).clip(-3, 3)
        vol_bonus = 1 / (1 + np.exp(-zv)) - 0.5  # -0.5〜+0.5
        comp = comp + volume_factor_weight * vol_bonus
    return comp


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_largecap.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERR] config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # ユニバースを決定（自動/手動）
    ids = resolve_universe(cfg)

    # BTCとの差分を計算するので、念のため bitcoin を含める（未含有なら追加）
    if "bitcoin" not in ids:
        ids.append("bitcoin")

    weights = cfg.get("weights", {})
    use_volume_factor = bool(cfg.get("use_volume_factor", True))
    volume_factor_weight = float(cfg.get("volume_factor_weight", 0.1))
    top_n = int(cfg.get("top_n", 10))
    out_csv = cfg.get("out_csv", "largecap_strength.csv")
    out_png = cfg.get("out_png", "largecap_strength.png")

    # 取得
    data = fetch_markets(ids, vs="usd")

    if not data:
        print("[ERR] empty response from CoinGecko", file=sys.stderr)
        sys.exit(2)

    rows = []
    for x in data:
        rows.append(
            {
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
            }
        )
    df = pd.DataFrame(rows)

    # BTC建て差分（BTC行が取れていなければ0埋め）
    btc_row = df.loc[df["id"] == "bitcoin"]
    if not btc_row.empty:
        for col in ["pct_1h", "pct_24h", "pct_7d"]:
            df[f"{col}_vs_btc"] = df[col] - float(btc_row[col].values[0])
    else:
        df["pct_1h_vs_btc"] = 0.0
        df["pct_24h_vs_btc"] = 0.0
        df["pct_7d_vs_btc"] = 0.0

    # スコア & ランク
    df["score"] = compute_scores(df, weights, use_volume_factor, volume_factor_weight)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    out_cols = [
        "rank", "name", "symbol", "price",
        "pct_1h", "pct_24h", "pct_7d",
        "pct_1h_vs_btc", "pct_24h_vs_btc", "pct_7d_vs_btc",
        "market_cap_rank", "market_cap", "volume_usd", "score",
    ]
    df[out_cols].to_csv(out_csv, index=False)

    # チャート
    top = df.head(top_n)
    plt.figure(figsize=(10, 6))
    plt.bar(top["symbol"], top["score"])
    plt.xticks(rotation=60, ha="right")    # ← シンボルを斜め表示
    plt.title("Large-Cap Relative Strength (Composite Score)")
    plt.xlabel("Symbol"); plt.ylabel("Score (z)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

    print(df[out_cols].head(top_n).to_string(index=False))
    print(f"\n[OK] wrote CSV: {out_csv}")
    print(f"[OK] wrote chart: {out_png}")


if __name__ == "__main__":
    main()
