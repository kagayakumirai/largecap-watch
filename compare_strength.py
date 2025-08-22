#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# USD建て × BTC建て 強弱を比較（ログ多め＆保存先固定）

import os, ast, re
import argparse, math, sys
from pathlib import Path
import requests, yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta
import numpy as np
from zoneinfo import ZoneInfo
import datetime as dt
import json
import pathlib

import time, random

def _get_json_with_retries(url, params, *, timeout=30, attempts=4):
    """
    CoinGecko向け: 429/5xx/ネットワークエラーを指数バックオフで再試行。
    envに API キーがあれば自動でヘッダ追加（任意）。
    """
    sess = requests.Session()
    headers = {"User-Agent": "largecap-watch/1.0"}
    # 任意: APIキー（ある場合のみ）
    if os.getenv("COINGECKO_API_KEY"):
        # 無料/デモ鍵: x-cg-demo-api-key / 有料鍵: x-cg-pro-api-key のどちらかを使ってください
        headers["x-cg-demo-api-key"] = os.environ["COINGECKO_API_KEY"]

    last_err = None
    for i in range(1, attempts + 1):
        try:
            r = sess.get(url, params=params, headers=headers, timeout=timeout)
            # レート制限
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
            # 2^i + 乱数（jitter）で待機
            wait = min(60, 2 ** i + random.uniform(0, 0.5))
            log(f"[RETRY] {e}; attempt {i}/{attempts}, sleep {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"GET failed after {attempts} attempts: {last_err}")


JST = ZoneInfo("Asia/Tokyo")
TRAILS_DIR = Path("data")
TRAILS_DIR.mkdir(parents=True, exist_ok=True)

def _now_jst_iso() -> str:
    return dt.datetime.now(JST).isoformat(timespec="seconds")

DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)
UNIVERSE_CACHE = DATA_DIR / "universe_cache.json"

# ---------- trails ----------
def persist_trails(side: str, df_now: pd.DataFrame, hours_keep: int = 168) -> Path:
    ts = _now_jst_iso()  # JSTで書く
    cur = df_now[["symbol", "score"]].copy()
    cur.insert(0, "ts", ts)
    cur.insert(1, "side", side)

    out = Path("data") / f"score_trails_{side}.csv"
    out.parent.mkdir(exist_ok=True, parents=True)
    header = not out.exists()
    cur.to_csv(out, index=False, mode="a", header=header)

    # 既存も含めて JST に統一しつつ掃除（古い行/重複を整理）
    try:
        hist = pd.read_csv(out)
        hist["ts"] = pd.to_datetime(hist["ts"], utc=True, errors="coerce").dt.tz_convert(JST)
        cutoff = pd.Timestamp.now(tz=JST) - pd.Timedelta(hours=hours_keep + 6)  # 余裕6h
        hist = hist[hist["ts"] >= cutoff]
        hist.sort_values(["ts", "symbol"], inplace=True)
        hist.drop_duplicates(["ts", "symbol"], keep="last", inplace=True)
        hist.to_csv(out, index=False)
    except Exception:
        pass

    return out

def load_trails(side: str):
    p = TRAILS_DIR / f"score_trails_{side}.csv"
    return None if not p.exists() else pd.read_csv(p, parse_dates=["ts"]).sort_values(["ts","symbol"])

# ---------- plot ----------
def _spread_labels(yvals, min_gap):
    """yvals（データ座標）の配列を、上下min_gap以上あくように順にずらす簡易アルゴリズム"""
    if not yvals: return yvals
    y = sorted(yvals)
    out = [y[0]]
    for v in y[1:]:
        out.append(max(v, out[-1] + min_gap))
    order = np.argsort(yvals)
    placed = {int(order[i]): out[i] for i in range(len(out))}
    return [placed[i] for i in range(len(yvals))]

def plot_trails(side: str, topn: int = 20, fname: str = None):
    hist = load_trails(side)
    if hist is None or hist.empty:
        print(f"[TRAILS] no data for {side}"); return None

    last_ts = hist["ts"].max()
    latest  = hist[hist["ts"] == last_ts].sort_values("score", ascending=False)
    keep    = latest["symbol"].head(topn).tolist()

    dfp = (hist[hist["symbol"].isin(keep)]
           .pivot(index="ts", columns="symbol", values="score")
           .sort_index())

    fig, ax = plt.subplots(figsize=(12, 5))

    # 直近の動き（Δ1）と現在値で“主役”を選ぶ
    if dfp.shape[0] >= 2:
        delta = (dfp.iloc[-1] - dfp.iloc[-2]).abs()
        movers = set(delta.nlargest(min(8, len(delta))).index)
    else:
        movers = set()
    top_now = set(dfp.iloc[-1].nlargest(min(8, dfp.shape[1])).index)
    label_cols = list(top_now | movers)

    for col in dfp.columns:
        series = dfp[col]
        if col in label_cols:
            ax.plot(series.index, series.values, linewidth=2.2, zorder=3)
        else:
            ax.plot(series.index, series.values, linewidth=1.2, alpha=0.35, zorder=2)

    ax.axhline(0, color="0.3", linewidth=1)

    x0, x1 = dfp.index[0], dfp.index[-1]
    pad = (x1 - x0) * 0.10
    ax.set_xlim(x0, x1 + pad)

    y_last_raw = [float(dfp[c].iloc[-1]) for c in label_cols]
    yr = ax.get_ylim(); min_gap = (yr[1] - yr[0]) * 0.03
    y_last = _spread_labels(y_last_raw, min_gap)

    for (col, y0, y_adj) in zip(label_cols, y_last_raw, y_last):
        dz = 0.0
        if dfp.shape[0] >= 2:
            dz = float(dfp[col].iloc[-1] - dfp[col].iloc[-2])
        arrow = "↑" if dz > 1e-3 else ("↓" if dz < -1e-3 else "→")
        ax.text(dfp.index[-1] + pad*0.02, y_adj,
                f"{col} {arrow} {dz:+.2f}",
                va="center", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=0.4),
                clip_on=False, zorder=5)
        ax.plot([dfp.index[-1]], [y0], marker="o", markersize=3, zorder=4)

    jst = timezone(timedelta(hours=9))
    title_ts = (dfp.index[-1].tz_convert(jst).strftime("%Y-%m-%d %H:%M JST")
                if getattr(dfp.index, "tz", None) else "")
    ax.set_title(f"{side.upper()}-score Trails (last 168h, top {topn})"
                 + (f" — {title_ts}" if title_ts else ""))
    ax.set_ylabel("Score (z)"); ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=28, ha="right")

    lim = float(max(abs(np.nanmin(dfp.values)), abs(np.nanmax(dfp.values)))) + 0.1
    ax.set_ylim(-lim, lim)

    fig.tight_layout()
    fname = fname or f"score_trails_{side}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[TRAILS] wrote {fname}")
    return fname

# ---------- universe ----------
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"

DEFAULT_EXCLUDE: set[str] = set()  # 追加の既定除外があればここへ

# symbol→id のフォールバック（必要に応じて拡張）
SYMBOL_TO_ID_FALLBACK = {
    "USDE":"ethena-usde","WETH":"weth","WEETH":"wrapped-eeth","WBETH":"wrapped-beacon-eth",
    "WSTETH":"wrapped-steth","STETH":"staked-ether",
    "USDT":"tether","USDC":"usd-coin","BUSD":"binance-usd","FDUSD":"first-digital-usd",
    "TUSD":"true-usd","PYUSD":"paypal-usd","WBTC":"wrapped-bitcoin",
    "LEO":"leo-token","BGB":"bitget-token","OKB":"okb","CRO":"crypto-com-chain",
    "KCS":"kucoin-shares","GT":"gatechain-token","HT":"huobi-token",
}

def _parse_listish(s: str | None) -> list[str]:
    if not s: return []
    s = s.strip()
    # 外側の一組のクォートを剥がす（"['A','B']" 対策）
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
    """
    symbolならフォールバック表でCoinGecko idへ、どちらにせよ小文字idを返す。
    """
    u = str(sym_or_id).upper()
    return (SYMBOL_TO_ID_FALLBACK.get(u, sym_or_id)).lower()

def fetch_top_mcap_ids(n: int = 12, exclude=None):
    # 除外集合は id 小文字で統一
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


def resolve_universe(cfg):
    # --- universe_mode を正規化 ---
    mode_raw = (cfg.get("universe_mode") or "manual").strip().lower()
    if mode_raw in {"top_mcap","top-mcap","top","mcap"}:
        mode = "top_mcap"
    else:
        # manual / universe / list などはすべて manual 扱い
        mode = "manual"
        print("[DBG] exclude has euro-coin?:", "euro-coin" in exclude_ids_lower)

    # なんでもリスト化する小道具
    def _as_list(x):
        if x is None: return []
        if isinstance(x, (list, tuple, set)): return list(x)
        return [x]

    # include
    include = [to_id_lower(x) for x in _as_list(cfg.get("include_ids"))]

    # env の除外（symbolでもidでも可）
    env_ex_syms = [x.upper() for x in _parse_listish(os.getenv("LARGECAP_EXCLUDE",""))]

    # config 側の exclude_ids / exclude を安全にマージ（nullや単体文字列を許容）
    ex_ids   = _as_list(cfg.get("exclude_ids"))
    ex_alias = _as_list(cfg.get("exclude"))          # 互換キーも許容
    ex_all   = env_ex_syms + ex_ids + ex_alias

    # 既定excludeも含めて、最終的に **id小文字** の集合へ
    exclude_ids_lower = {to_id_lower(x) for x in (set(DEFAULT_EXCLUDE) | set(ex_all))}

    if mode == "top_mcap":
        # env 優先 → cfg → 既定12
        n = int(os.getenv("LARGECAP_TOP") or cfg.get("top_mcap_n", 12))
        ids = fetch_top_mcap_ids(n=n, exclude=exclude_ids_lower)
        ids = [i for i in ids if i.lower() not in exclude_ids_lower]
        # include を先頭にマージ（重複排除）
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

# ---------- fetch ----------
CG_URL = "https://api.coingecko.com/api/v3/coins/markets"

def log(msg): print(f"[LOG] {msg}")

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


# ---------- scoring ----------
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

# ---------- main ----------
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

    # trails
    latest_usd = df[["symbol","usd_score"]].rename(columns={"usd_score":"score"})
    latest_btc = df[["symbol","btc_score"]].rename(columns={"btc_score":"score"})
    persist_trails("usd", latest_usd)
    persist_trails("btc", latest_btc)

    plot_trails("usd", topn=20, fname="score_trails_usd.png")
    plot_trails("btc", topn=20, fname="score_trails_btc.png")

    out_csv = out_dir / cfg.get("out_csv", "largecap_compare.csv")
    out_png = out_dir / cfg.get("out_png_scatter", "largecap_usd_vs_btc.png")

    out_cols = ["name","symbol","market_cap_rank","usd_score","btc_score","quadrant",
                "pct_1h","pct_24h","pct_7d","pct_1h_btc","pct_24h_btc","pct_7d_btc",
                "price_usd","price_btc","market_cap","volume_usd"]
    df.sort_values(["btc_score","usd_score"], ascending=False)[out_cols].to_csv(out_csv, index=False)
    log(f"wrote CSV: {out_csv}")

    # ==== compare フィルタ＆スコア ====
    src_topn  = int(cfg.get("compare_source_topn", 30))   # 片側で参照するTopN
    cmp_topn  = int(cfg.get("compare_top_n", 20))         # 最終表示数
    guard_min = float(cfg.get("compare_guard_min", 0.0))  # 両軸の最低ライン(z)
    lam       = float(cfg.get("compare_penalty_lambda", 0.3))  # 不均衡ペナルティ
    sort_mode = str(cfg.get("compare_sort", "sum")).lower()

    sym_col = "symbol" if "symbol" in df.columns else None
    original_df = df.set_index(sym_col, drop=False).copy() if sym_col else df.copy()

    # ランク計算
    df["rank_usd"] = df["usd_score"].rank(ascending=False, method="first")
    df["rank_btc"] = df["btc_score"].rank(ascending=False, method="first")

    # OR＋ガード（片側TopN以内 かつ 両軸 guard_min 以上）
    mask_or    = (df["rank_usd"] <= src_topn) | (df["rank_btc"] <= src_topn)
    mask_guard = (df["usd_score"] >= guard_min) & (df["btc_score"] >= guard_min)
    df = df.loc[mask_or & mask_guard].copy()

    # 並べ替えスコア
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

    # 上位 compare_top_n だけ残す（保険：空なら合計で埋める）
    if df.empty:
        df = original_df.copy()
        df["score"] = df["usd_score"] + df["btc_score"]
    df = df.sort_values("score", ascending=False).head(cmp_topn)

    # ⚠ 足りないときのフォールバック
    if len(df) < cmp_topn:
        need = cmp_topn - len(df)
        rest = original_df.loc[~original_df.index.isin(df[sym_col])] if sym_col else original_df.drop(df.index, errors="ignore")
        rest = rest.copy()
        rest["score"] = rest["usd_score"] + rest["btc_score"]
        add = rest.nlargest(need, "score")
        df = pd.concat([df, add], ignore_index=True)

    # 散布図描画
    plt.figure(figsize=(8,6))
    plt.scatter(df["usd_score"], df["btc_score"])
    plt.axvline(th_u); plt.axhline(th_b)
    label_fs = cfg.get("label_fontsize",7)
    for _, r in df.iterrows():
        plt.annotate(r["symbol"], (r["usd_score"], r["btc_score"]),
                     xytext=(4,4), textcoords="offset points", fontsize=label_fs)
    plt.title("USD-score vs BTC-score (Large-Cap)")
    plt.xlabel("USD composite (z)"); plt.ylabel("BTC-quoted composite (z)")
    plt.tight_layout()

    jst = timezone(timedelta(hours=9))
    stamp = datetime.now(jst).strftime("%Y-%m-%d %H:%M")
    ax = plt.gca()
    ax.set_title(f"USD-score vs BTC-score (Large-Cap) — {stamp} JST")
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    ax.text(x1*0.98, y1*0.98, "Bull / Bull", ha="right", va="top", fontsize=10, alpha=0.7)
    ax.text(x0*0.02, y1*0.98, "Bear / Bull", ha="left",  va="top", fontsize=10, alpha=0.7)
    ax.text(x1*0.98, y0*0.02, "Bull / Bear", ha="right", va="bottom", fontsize=10, alpha=0.7)
    ax.text(x0*0.02, y0*0.02, "Bear / Bear", ha="left",  va="bottom", fontsize=10, alpha=0.7)

    plt.savefig(out_png, dpi=150)
    plt.close()
    log(f"wrote PNG: {out_png}")

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


