#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# USD建て × BTC建て 強弱を比較（ログ多め＆保存先固定）
# 2025/9/7 compare は “計算＆散布図＆CSV” だけ。トレイル画像は別スクリプトで描画。

import os, ast, re, math, sys, time, random, json
import argparse
from pathlib import Path
from datetime import datetime

import requests
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
TRAILS_DIR = Path("data")
TRAILS_DIR.mkdir(parents=True, exist_ok=True)
TRAILS_DB = TRAILS_DIR / "trails_db.csv"

# ------------------------------------------------------------
# 共通ユーティリティ
# ------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[LOG] {msg}")

def _get_json_with_retries(url, params, *, timeout=30, attempts=4):
    """429/5xx/通信失敗を指数バックオフで再試行。APIキーがあれば自動付加。"""
    sess = requests.Session()
    headers = {"User-Agent": "largecap-watch/1.0"}
    if os.getenv("COINGECKO_API_KEY"):
        headers["x-cg-demo-api-key"] = os.environ["COINGECKO_API_KEY"]  # 無料/デモ鍵

    last_err = None
    for i in range(1, attempts + 1):
        try:
            r = sess.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 429:
                wait = r.headers.get("Retry-After")
                wait = int(wait) if str(wait).isdigit() else 2 ** i
                log(f"[HTTP429] rate-limited; sleep {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            if i == attempts:
                break
            wait = min(60, 2 ** i + random.uniform(0, 0.5))
            log(f"[RETRY] {e}; attempt {i}/{attempts}, sleep {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"GET failed after {attempts} attempts: {last_err}")

# ------------------------------------------------------------
# trails_db.csv へ upsert（UTC tz-aware で統一）
# ------------------------------------------------------------
def upsert_trails_db(df_scores: pd.DataFrame, hours_keep: int = 24 * 45) -> Path:
    """
    df_scores: 必須列 ['symbol','usd_score','btc_score']
    同一 (timestamp, symbol) は最後を採用。保持期間は hours_keep（既定=45日）。
    すべて tz-aware(UTC) に統一。
    """
    TRAILS_DIR.mkdir(parents=True, exist_ok=True)

    # 15分バケツに揃えた UTC の tz-aware 時刻
    run_ts = pd.Timestamp.now(tz="UTC").floor("15min")

    # 追記行（UTC付きタイムスタンプ）
    add = df_scores[["symbol", "usd_score", "btc_score"]].copy()
    add.insert(0, "timestamp", run_ts)

    # 既存CSV → timestamp を UTC tz-aware に正規化（format は固定しない）
    if TRAILS_DB.exists():
        hist = pd.read_csv(TRAILS_DB, dtype={"symbol": "string"})
        hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True, errors="coerce")
    else:
        hist = pd.DataFrame(columns=["timestamp", "symbol", "usd_score", "btc_score"])

    # 連結 → 型そろえ
    merged = pd.concat([hist, add], ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
    merged["symbol"] = merged["symbol"].astype("string")
    merged["usd_score"] = pd.to_numeric(merged["usd_score"], errors="coerce")
    merged["btc_score"] = pd.to_numeric(merged["btc_score"], errors="coerce")
    merged = merged.dropna(subset=["timestamp", "symbol"])

    # 保持期間で間引き（両者とも tz-aware なので比較OK）
    cutoff = run_ts - pd.Timedelta(hours=hours_keep)
    merged = merged[merged["timestamp"] >= cutoff]

    # 同一 (timestamp, symbol) は最後を残す
    merged = merged.sort_values(["timestamp", "symbol"])
    merged = merged.drop_duplicates(["timestamp", "symbol"], keep="last")

    # 保存
    merged.to_csv(TRAILS_DB, index=False)
    print(f"[TRAILS] upserted into {TRAILS_DB}, last={merged['timestamp'].max()}")
    return TRAILS_DB

# ------------------------------------------------------------
# universe resolve / fetch / scoring
# ------------------------------------------------------------
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
DEFAULT_EXCLUDE: set[str] = set()
SYMBOL_TO_ID_FALLBACK = {
    "USDE":"ethena-usde","WETH":"weth","WEETH":"wrapped-eeth","WBETH":"wrapped-beacon-eth",
    "WSTETH":"wrapped-steth","STETH":"staked-ether","USDT":"tether","USDC":"usd-coin",
    "BUSD":"binance-usd","FDUSD":"first-digital-usd","TUSD":"true-usd","PYUSD":"paypal-usd",
    "WBTC":"wrapped-bitcoin",
}

def _parse_listish(s: str | None) -> list[str]:
    if not s:
        return []
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        try:
            return [str(x).strip() for x in ast.literal_eval(s)]
        except Exception:
            pass
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [x for x in re.split(r"\s+", s) if x]

def to_id_lower(sym_or_id: str) -> str:
    u = str(sym_or_id).upper()
    return (SYMBOL_TO_ID_FALLBACK.get(u, sym_or_id)).lower()

def fetch_top_mcap_ids(n: int = 12, exclude=None) -> list[str]:
    exclude_set = {str(x).lower() for x in (exclude or set())}
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 250,
        "page": 1,
        "sparkline": "false",
    }
    rows = _get_json_with_retries(COINGECKO_URL, params, timeout=30, attempts=4)
    ids = []
    for x in rows:
        cid = str(x.get("id") or "").lower()
        if not cid or cid in exclude_set:
            continue
        ids.append(cid)
        if len(ids) >= n:
            break
    return ids

def resolve_universe(cfg) -> list[str]:
    mode_raw = (cfg.get("universe_mode") or "manual").strip().lower()
    mode = "top_mcap" if mode_raw in {"top_mcap", "top-mcap", "top", "mcap"} else "manual"

    def _as_list(x):
        return [] if x is None else (list(x) if isinstance(x, (list, tuple, set)) else [x])

    include = [to_id_lower(x) for x in _as_list(cfg.get("include_ids"))]
    env_ex_syms = [x.upper() for x in _parse_listish(os.getenv("LARGECAP_EXCLUDE", ""))]
    ex_ids = _as_list(cfg.get("exclude_ids"))
    ex_alias = _as_list(cfg.get("exclude"))
    ex_all = env_ex_syms + ex_ids + ex_alias
    exclude_ids_lower = {to_id_lower(x) for x in (set(DEFAULT_EXCLUDE) | set(ex_all))}

    if mode == "top_mcap":
        n = int(os.getenv("LARGECAP_TOP") or cfg.get("top_mcap_n", 12))
        ids = fetch_top_mcap_ids(n=n, exclude=exclude_ids_lower)
        ids = [i for i in ids if i.lower() not in exclude_ids_lower]
        ids = list(dict.fromkeys(include + ids))
        print(f"[DBG] universe_mode=top_mcap, n={n}, resolved={len(ids)}")
        return ids
    else:
        ids = [to_id_lower(x) for x in _as_list(cfg.get("universe_ids"))]
        if not ids:
            raise ValueError("manual モードでは universe_ids を配列で指定してください（CoinGecko id）")
        ids = list(dict.fromkeys(include + ids))
        ids = [c for c in ids if c.lower() not in exclude_ids_lower]
        print(f"[DBG] universe_mode=manual, resolved={len(ids)}")
        return ids

CG_URL = "https://api.coingecko.com/api/v3/coins/markets"

def fetch(ids, vs):
    log(f"fetch start vs={vs}, ids={len(ids)}")
    js = _get_json_with_retries(
        CG_URL,
        {
            "vs_currency": vs,
            "ids": ",".join(ids),
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d",
        },
        timeout=30,
        attempts=4,
    )
    log(f"fetch done vs={vs}, rows={len(js)}")
    return js

def zscore(s):
    s = pd.Series(s, dtype="float64")
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or math.isnan(sd):
        return pd.Series([0] * len(s), index=s.index)
    return (s - mu) / sd

def score_usd(df, w, use_vol, vol_w):
    from numpy import exp
    comp = (
        w.get("pct_1h", 0) * zscore(df["pct_1h"])
        + w.get("pct_24h", 0) * zscore(df["pct_24h"])
        + w.get("pct_7d", 0) * zscore(df["pct_7d"])
    )
    if use_vol:
        zv = zscore(df["volume_usd"]).clip(-3, 3)
        comp = comp + vol_w * (1 / (1 + exp(-zv)) - 0.5)
    return comp

def score_btc(df, w, use_vol, vol_w):
    from numpy import exp
    comp = (
        w.get("pct_1h_btc", 0) * zscore(df["pct_1h_btc"])
        + w.get("pct_24h_btc", 0) * zscore(df["pct_24h_btc"])
        + w.get("pct_7d_btc", 0) * zscore(df["pct_7d_btc"])
    )
    if use_vol:
        zv = zscore(df["volume_usd"]).clip(-3, 3)
        comp = comp + vol_w * (1 / (1 + exp(-zv)) - 0.5)
    return comp

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    print("== compare_strength starting ==")

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

    rows_usd = [
        {
            "id": x["id"],
            "symbol": str(x.get("symbol", "")).upper(),
            "name": x.get("name"),
            "price_usd": x.get("current_price"),
            "pct_1h": x.get("price_change_percentage_1h_in_currency") or 0.0,
            "pct_24h": x.get("price_change_percentage_24h_in_currency") or 0.0,
            "pct_7d": x.get("price_change_percentage_7d_in_currency") or 0.0,
            "volume_usd": x.get("total_volume"),
            "market_cap": x.get("market_cap"),
            "market_cap_rank": x.get("market_cap_rank"),
        }
        for x in usd
    ]
    df_usd = pd.DataFrame(rows_usd)
    log(f"df_usd shape={df_usd.shape}")

    rows_btc = [
        {
            "id": x["id"],
            "price_btc": x.get("current_price"),
            "pct_1h_btc": x.get("price_change_percentage_1h_in_currency") or 0.0,
            "pct_24h_btc": x.get("price_change_percentage_24h_in_currency") or 0.0,
            "pct_7d_btc": x.get("price_change_percentage_7d_in_currency") or 0.0,
        }
        for x in btc
    ]
    df_btc = pd.DataFrame(rows_btc)
    log(f"df_btc shape={df_btc.shape}")

    df = df_usd.merge(df_btc, on="id", how="inner")
    log(f"merged shape={df.shape}")

    df["usd_score"] = score_usd(
        df,
        cfg["weights_usd"],
        bool(cfg.get("use_volume_factor_usd", True)),
        float(cfg.get("volume_factor_weight_usd", 0.1)),
    )
    df["btc_score"] = score_btc(
        df,
        cfg["weights_btc"],
        bool(cfg.get("use_volume_factor_btc", True)),
        float(cfg.get("volume_factor_weight_btc", 0.1)),
    )

    th_u = float(cfg.get("threshold_usd", 0.0))
    th_b = float(cfg.get("threshold_btc", 0.0))

    def label(u, b):
        if u >= th_u and b >= th_b:
            return "A 強×強（本命）"
        if u >= th_u and b < th_b:
            return "B 強×弱（USD強・BTC優位）"
        if u < th_u and b >= th_b:
            return "C 弱×強（BTC重・アルト優勢）"
        return "D 弱×弱（様子見）"

    df["quadrant"] = [label(u, b) for u, b in zip(df["usd_score"], df["btc_score"])]

    # ---- Trails 追記（統合：trails_db.csv を正とする）
    scores_for_trails = df[["symbol", "usd_score", "btc_score"]].copy()
    upsert_trails_db(scores_for_trails, hours_keep=24 * 45)

    # ---- 出力
    out_csv = out_dir / cfg.get("out_csv", "largecap_compare.csv")
    out_png = out_dir / cfg.get("out_png_scatter", "largecap_usd_vs_btc.png")

    out_cols = [
        "name",
        "symbol",
        "market_cap_rank",
        "usd_score",
        "btc_score",
        "quadrant",
        "pct_1h",
        "pct_24h",
        "pct_7d",
        "pct_1h_btc",
        "pct_24h_btc",
        "pct_7d_btc",
        "price_usd",
        "price_btc",
        "market_cap",
        "volume_usd",
    ]
    df.sort_values(["btc_score", "usd_score"], ascending=False)[out_cols].to_csv(
        out_csv, index=False
    )
    log(f"wrote CSV: {out_csv}")

    # ==== compare フィルタ＆スコア ====
    src_topn = int(cfg.get("compare_source_topn", 30))
    cmp_topn = int(cfg.get("compare_top_n", 20))
    guard_min = float(cfg.get("compare_guard_min", 0.0))
    lam = float(cfg.get("compare_penalty_lambda", 0.3))
    sort_mode = str(cfg.get("compare_sort", "sum")).lower()

    sym_col = "symbol" if "symbol" in df.columns else None
    original_df = df.set_index(sym_col, drop=False).copy() if sym_col else df.copy()

    df["rank_usd"] = df["usd_score"].rank(ascending=False, method="first")
    df["rank_btc"] = df["btc_score"].rank(ascending=False, method="first")

    mask_or = (df["rank_usd"] <= src_topn) | (df["rank_btc"] <= src_topn)
    mask_guard = (df["usd_score"] >= guard_min) & (df["btc_score"] >= guard_min)
    df = df.loc[mask_or & mask_guard].copy()

    if sort_mode == "sum":
        base = df["usd_score"] + df["btc_score"]
        penalty = lam * (df["usd_score"] - df["btc_score"]).abs()
        df["score"] = base - penalty
    elif sort_mode == "abs":
        df["score"] = df["usd_score"].abs() + df["btc_score"].abs()
    elif sort_mode == "usd":
        df["score"] = df["usd_score"]
    elif sort_mode == "btc":
        df["score"] = df["btc_score"]
    else:
        df["score"] = df["usd_score"] + df["btc_score"]

    if df.empty:
        df = original_df.copy()
        df["score"] = df["usd_score"] + df["btc_score"]
    df = df.sort_values("score", ascending=False).head(cmp_topn)

    if len(df) < cmp_topn:
        need = cmp_topn - len(df)
        rest = (
            original_df.loc[~original_df.index.isin(df[sym_col])]
            if sym_col
            else original_df.drop(df.index, errors="ignore")
        )
        rest = rest.copy()
        rest["score"] = rest["usd_score"] + rest["btc_score"]
        add = rest.nlargest(need, "score")
        df = pd.concat([df, add], ignore_index=True)

    # 散布図描画（JST）
    plt.figure(figsize=(8, 6))
    plt.scatter(df["usd_score"], df["btc_score"])
    plt.axvline(th_u)
    plt.axhline(th_b)
    label_fs = cfg.get("label_fontsize", 7)
    for _, r in df.iterrows():
        plt.annotate(
            r["symbol"],
            (r["usd_score"], r["btc_score"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=label_fs,
        )

    jst_now = datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")
    ax = plt.gca()
    ax.set_title(f"USD-score vs BTC-score (Large-Cap) — {jst_now}")
    ax.set_xlabel("USD composite (z)")
    ax.set_ylabel("BTC-quoted composite (z)")
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.text(x1 * 0.98, y1 * 0.98, "Bull / Bull", ha="right", va="top", fontsize=6, alpha=0.7)
    ax.text(x0 * 0.02, y1 * 0.98, "Bear / Bull", ha="left", va="top", fontsize=6, alpha=0.7)
    ax.text(x1 * 0.98, y0 * 0.02, "Bull / Bear", ha="right", va="bottom", fontsize=6, alpha=0.7)
    ax.text(x0 * 0.02, y0 * 0.02, "Bear / Bear", ha="left", va="bottom", fontsize=6, alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    log(f"wrote PNG: {out_png}")

    show = df.sort_values(["btc_score", "usd_score"], ascending=False)[
        ["symbol", "usd_score", "btc_score", "quadrant"]
    ]
    print(show.to_string(index=False))
    print("[DONE]")
    print("before filter:", len(df))
    print("OR kept:", ((df["rank_usd"] <= src_topn) | (df["rank_btc"] <= src_topn)).sum())
    if guard_min is not None:
        print(
            f"guard_min={guard_min} kept:",
            ((df["usd_score"] >= guard_min) & (df["btc_score"] >= guard_min)).sum(),
        )

if __name__ == "__main__":
    main()
