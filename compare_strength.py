#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# USD建て × BTC建て 強弱を比較（ログ多め＆保存先固定）

import argparse, math
from pathlib import Path
import requests, yaml
import pandas as pd
import matplotlib.pyplot as plt
import json, time, pathlib, requests

DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)
UNIVERSE_CACHE = DATA_DIR / "universe_cache.json"
import requests, yaml, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import math, sys

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"

DEFAULT_EXCLUDE = {
}

def fetch_top_mcap_ids(n=12, exclude=None):
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
        if not cid or cid in exclude:
            continue
        ids.append(cid)
        if len(ids) >= n:
            break
    return ids

def resolve_universe(cfg):
    mode = (cfg.get("universe_mode") or "manual").lower()
    exclude = set(DEFAULT_EXCLUDE) | set(cfg.get("exclude_ids", []))
    include = list(cfg.get("include_ids", []))

    if mode == "top_mcap":
        n = int(cfg.get("top_mcap_n", 12))
        ids = fetch_top_mcap_ids(n=n, exclude=exclude)
        return list(dict.fromkeys(include + ids))  # include を先頭にマージ
    else:
        ids = cfg.get("universe_ids")
        if not isinstance(ids, (list, tuple)) or not ids:
            raise ValueError("manual モードでは universe_ids を配列で指定してください")
        ids = list(dict.fromkeys(include + list(ids)))
        return [c for c in ids if c not in exclude]




CG_MARKETS = "https://api.coingecko.com/api/v3/coins/markets"
CG_URL = "https://api.coingecko.com/api/v3/coins/markets"

def log(msg): print(f"[LOG] {msg}")

def fetch(ids, vs):
    log(f"fetch start vs={vs}, ids={len(ids)}")
    r = requests.get(
        CG_URL,
        params={
            "vs_currency": vs,
            "ids": ",".join(ids),
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d",
        },
        timeout=30,
    )
    log(f"status={r.status_code}")
    r.raise_for_status()
    js = r.json()
    log(f"fetch done vs={vs}, rows={len(js)}")
    return js

def zscore(s):
    s = pd.Series(s, dtype="float64")
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or math.isnan(sd): return pd.Series([0]*len(s), index=s.index)
    return (s - mu) / sd

def score_usd(df, w, use_vol, vol_w):
    from numpy import exp
    comp = (w.get("pct_1h",0)*zscore(df["pct_1h"]) +
            w.get("pct_24h",0)*zscore(df["pct_24h"]) +
            w.get("pct_7d",0)*zscore(df["pct_7d"]))
    if use_vol:
        zv = zscore(df["volume_usd"]).clip(-3,3)
        comp = comp + vol_w*(1/(1+exp(-zv)) - 0.5)
    return comp

def score_btc(df, w, use_vol, vol_w):
    from numpy import exp
    comp = (w.get("pct_1h_btc",0)*zscore(df["pct_1h_btc"]) +
            w.get("pct_24h_btc",0)*zscore(df["pct_24h_btc"]) +
            w.get("pct_7d_btc",0)*zscore(df["pct_7d_btc"]))
    if use_vol:
        zv = zscore(df["volume_usd"]).clip(-3,3)
        comp = comp + vol_w*(1/(1+exp(-zv)) - 0.5)
    return comp

def main():
    print("== compare_strength starting ==")  # バナー

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_largecap_compare.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    log(f"config path={cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out_dir = cfg_path.parent
    log(f"out_dir={out_dir}")

    ids = resolve_universe(cfg)

    try:
        usd = fetch(ids, "usd")
        btc = fetch(ids, "btc")
    except Exception as e:
        print("[ERROR] fetch failed:", e)
        return

    rows_usd = [{
        "id": x["id"],
        "symbol": str(x.get("symbol","")).upper(),
        "name": x.get("name"),
        "price_usd": x.get("current_price"),
        "pct_1h": x.get("price_change_percentage_1h_in_currency") or 0.0,
        "pct_24h": x.get("price_change_percentage_24h_in_currency") or 0.0,
        "pct_7d": x.get("price_change_percentage_7d_in_currency") or 0.0,
        "volume_usd": x.get("total_volume"),
        "market_cap": x.get("market_cap"),
        "market_cap_rank": x.get("market_cap_rank"),
    } for x in usd]
    df_usd = pd.DataFrame(rows_usd); log(f"df_usd shape={df_usd.shape}")

    rows_btc = [{
        "id": x["id"],
        "price_btc": x.get("current_price"),
        "pct_1h_btc": x.get("price_change_percentage_1h_in_currency") or 0.0,
        "pct_24h_btc": x.get("price_change_percentage_24h_in_currency") or 0.0,
        "pct_7d_btc": x.get("price_change_percentage_7d_in_currency") or 0.0,
    } for x in btc]
    df_btc = pd.DataFrame(rows_btc); log(f"df_btc shape={df_btc.shape}")

    df = df_usd.merge(df_btc, on="id", how="inner"); log(f"merged shape={df.shape}")

    df["usd_score"] = score_usd(
        df, cfg["weights_usd"],
        bool(cfg.get("use_volume_factor_usd", True)),
        float(cfg.get("volume_factor_weight_usd", 0.1)),
    )
    df["btc_score"] = score_btc(
        df, cfg["weights_btc"],
        bool(cfg.get("use_volume_factor_btc", True)),
        float(cfg.get("volume_factor_weight_btc", 0.1)),
    )

    th_u = float(cfg.get("threshold_usd", 0.0))
    th_b = float(cfg.get("threshold_btc", 0.0))
    def label(u, b):
        if u >= th_u and b >= th_b:   return "A 強×強（本命）"
        if u >= th_u and b <  th_b:   return "B 強×弱（USD強・BTC優位）"
        if u <  th_u and b >= th_b:   return "C 弱×強（BTC重・アルト優勢）"
        return "D 弱×弱（様子見）"
    df["quadrant"] = [label(u,b) for u,b in zip(df["usd_score"], df["btc_score"])]

    out_csv = out_dir / cfg.get("out_csv", "largecap_compare.csv")
    out_png = out_dir / cfg.get("out_png_scatter", "largecap_usd_vs_btc.png")

    out_cols = ["name","symbol","market_cap_rank","usd_score","btc_score","quadrant",
                "pct_1h","pct_24h","pct_7d","pct_1h_btc","pct_24h_btc","pct_7d_btc",
                "price_usd","price_btc","market_cap","volume_usd"]
    df.sort_values(["btc_score","usd_score"], ascending=False)[out_cols].to_csv(out_csv, index=False)
    log(f"wrote CSV: {out_csv}")

    # ==== compare フィルタ＆スコア（取りこぼし回避版） ====
    src_topn  = int(cfg.get("compare_source_topn", 30))   # 片側で参照するTopN
    cmp_topn  = int(cfg.get("compare_top_n", 20))         # 最終表示数
    guard_min = float(cfg.get("compare_guard_min", 0.0))  # 両軸の最低ライン(z)
    lam       = float(cfg.get("compare_penalty_lambda", 0.3))  # 不均衡ペナルティ

    # ランク計算
    df["rank_usd"] = df["usd_score"].rank(ascending=False, method="first")
    df["rank_btc"] = df["btc_score"].rank(ascending=False, method="first")

    # OR＋ガード（片側TopN以内 かつ 両軸 guard_min 以上）
    mask_or    = (df["rank_usd"] <= src_topn) | (df["rank_btc"] <= src_topn)
    mask_guard = (df["usd_score"] >= guard_min) & (df["btc_score"] >= guard_min)
    df = df.loc[mask_or & mask_guard].copy()

    # スコア = 合計 − λ×不均衡
    df["score"] = (df["usd_score"] + df["btc_score"]) - lam * (df["usd_score"] - df["btc_score"]).abs()

    # 上位 compare_top_n だけ残す（保険：空なら合計で埋める）
    if df.empty:
        df = backup_df.copy() if "backup_df" in locals() else original_df.copy()  # 任意
        df["score"] = df["usd_score"] + df["btc_score"]
    df = df.sort_values("score", ascending=False).head(cmp_topn)
    
    # フィルタ後 df が少ないときは合計スコアで埋める
    if len(df) < cmp_topn:
        need = cmp_topn - len(df)
        rest = original_df.drop(df.index).copy()
        rest["score"] = rest["usd_score"] + rest["btc_score"]
        df = pd.concat([df, rest.nlargest(need, "score")], ignore_index=True)

    # 散布図描画
    plt.figure(figsize=(8,6))
    plt.scatter(df["usd_score"], df["btc_score"])
    plt.axvline(th_u); plt.axhline(th_b)
    label_fs = cfg.get("label_fontsize",7)
    for _, r in df.iterrows():
        plt.annotate(r["symbol"], (r["usd_score"], r["btc_score"]),
                     xytext=(4,4), textcoords="offset points",fontsize=label_fs)
    plt.title("USD-score vs BTC-score (Large-Cap)")
    plt.xlabel("USD composite (z)"); plt.ylabel("BTC-quoted composite (z)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    log(f"wrote PNG: {out_png}")

    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    ax.text(x1*0.98, y1*0.98, "強気/強気", ha="right", va="top", alpha=0.7, fontsize=10)
    ax.text(x0*0.02, y1*0.98, "弱気/強気", ha="left",  va="top", alpha=0.7, fontsize=10)
    ax.text(x1*0.98, y0*0.02, "強気/弱気", ha="right", va="bottom", alpha=0.7, fontsize=10)
    ax.text(x0*0.02, y0*0.02, "弱気/弱気", ha="left",  va="bottom", alpha=0.7, fontsize=10)

    from datetime import datetime, timezone, timedelta
    jst = timezone(timedelta(hours=9))
    date = datetime.now(jst).strftime("%Y-%m-%d")  # 例: 2025-08-21
    ax.set_title(f"USD-score vs BTC-score (Large-Cap) — {date}")
    # そのまま savefig へ


    show = df.sort_values(["btc_score","usd_score"], ascending=False)[["symbol","usd_score","btc_score","quadrant"]]
    print(show.to_string(index=False))
    print("[DONE]")

    print("before filter:", len(df))
    print("OR kept:", ((df["rank_usd"]<=src_topn)|(df["rank_btc"]<=src_topn)).sum())
    if guard_min is not None:
        print(f"guard_min={guard_min} kept:",
              ((df["usd_score"]>=guard_min)&(df["btc_score"]>=guard_min)).sum())

if __name__ == "__main__":
    main()












