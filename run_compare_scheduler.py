#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_strength.py を一定間隔で実行し、
- 通常の PNG/CSV を保存 & （任意で）Discord送信
- data/score_trails_*.csv を読み、JSTでスコア推移グラフ(score_trails_*.png)を作成
"""

import argparse
import subprocess
import time
import datetime as dt
import sys
import shutil
from pathlib import Path
from typing import List

import yaml
import json
import pandas as pd

try:
    import requests  # optional（Discord 送信用）
except Exception:
    requests = None

# ===== タイムゾーン（JST） =====
from zoneinfo import ZoneInfo
JST = ZoneInfo("Asia/Tokyo")

# ---------- helpers ----------
def now_hm() -> str:
    return dt.datetime.now(JST).strftime("%Y%m%d_%H%M")

def now_human() -> str:
    return dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")

def run_py(python_exe: str, script_path: Path, *args) -> bool:
    cmd = [python_exe, str(script_path), *[str(a) for a in args]]
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return proc.returncode == 0

def summarize(csv_path: Path, top_n: int = 5) -> str:
    try:
        df = pd.read_csv(csv_path)
        # 想定列: symbol, usd_score, btc_score, quadrant
        df_sorted = df.sort_values(["btc_score", "usd_score"], ascending=False)
        top = df_sorted.head(top_n)[["symbol", "usd_score", "btc_score", "quadrant"]]
        counts = df["quadrant"].value_counts().to_dict()
        lines = [f"Top{top_n} (btc→usd):"]
        for _, r in top.iterrows():
            lines.append(
                f"- {r['symbol']}: usd={r['usd_score']:.2f}, btc={r['btc_score']:.2f}, {r['quadrant']}"
            )
        qorder = ["A 強×強（本命）", "B 強×弱（USD強・BTC優位）", "C 弱×強（BTC重・アルト優勢）", "D 弱×弱（様子見）"]
        lines.append("Quadrants: " + " / ".join([f"{k}:{counts.get(k,0)}" for k in qorder]))
        return "\n".join(lines)
    except Exception as e:
        return f"(summary failed: {e})"

# ===== Trails 読み込み & 描画（JST） =====
TRAILS_DIR = Path("data")
TRAILS_DIR.mkdir(exist_ok=True)

def load_trails(side: str) -> pd.DataFrame | None:
    """
    side: 'usd' or 'btc'
    CSV: data/score_trails_<side>.csv with columns [ts, side, symbol, score]
    ts は tz付き（推奨）。tzなしでもJSTに合わせる。
    """
    p = TRAILS_DIR / f"score_trails_{side}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["ts"])
    # ts を必ず JST に統一
    try:
        if getattr(df["ts"].dt, "tz", None) is None:
            # tz なし → JST として解釈
            df["ts"] = df["ts"].dt.tz_localize(JST)
        else:
            df["ts"] = df["ts"].dt.tz_convert(JST)
    except Exception:
        # 文字列などフォールバック（UTC前提→JST）
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(JST)
    df.sort_values(["ts", "symbol"], inplace=True)
    return df

def plot_trails(side: str, hours: int = 168, topn: int = 20, smooth: int = 1, out_png: Path | None = None) -> Path | None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    hist = load_trails(side)
    if hist is None or hist.empty:
        print(f"[TRAILS] no data for {side}")
        return None

    # 時間窓で絞る（JST基準）
    cutoff = pd.Timestamp.now(tz=JST) - pd.Timedelta(hours=hours)
    hist = hist[hist["ts"] >= cutoff]
    if hist.empty:
        print(f"[TRAILS] window empty for {side}")
        return None

    # 最新時点での上位 topn を採用
    last_ts = hist["ts"].max()
    latest = hist[hist["ts"] == last_ts].sort_values("score", ascending=False)
    keep = latest["symbol"].head(topn).tolist()

    dfp = (
        hist[hist["symbol"].isin(keep)]
        .pivot(index="ts", columns="symbol", values="score")
        .sort_index()
    )

    # 平滑化（任意）
    if smooth and smooth > 1:
        dfp = dfp.rolling(smooth, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    # “動いた銘柄”を太め表示
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

    # y=0
    ax.axhline(0, color="0.35", linewidth=1)

    # 右側に余白
    x0, x1 = dfp.index[0], dfp.index[-1]
    pad = (x1 - x0) * 0.10
    ax.set_xlim(x0, x1 + pad)

    # 右端ラベル（重なり回避の簡易ずらし）
    def _spread_labels(yvals: List[float], min_gap: float) -> List[float]:
        if not yvals: return yvals
        y_sorted = sorted(yvals)
        out = [y_sorted[0]]
        for v in y_sorted[1:]:
            out.append(max(v, out[-1] + min_gap))
        order = np.argsort(yvals)
        placed = {int(order[i]): out[i] for i in range(len(out))}
        return [placed[i] for i in range(len(yvals))]

    y_last_raw = [float(dfp[c].iloc[-1]) for c in label_cols]
    yr = (float(np.nanmin(dfp.values)), float(np.nanmax(dfp.values)))
    min_gap = (yr[1] - yr[0]) * 0.03 if yr[1] > yr[0] else 0.05
    y_last = _spread_labels(y_last_raw, min_gap)

    for (col, y0_, y_adj) in zip(label_cols, y_last_raw, y_last):
        dz = float(dfp[col].iloc[-1] - (dfp[col].iloc[-2] if dfp.shape[0] >= 2 else dfp[col].iloc[-1]))
        arrow = "↑" if dz > 1e-3 else ("↓" if dz < -1e-3 else "→")
        ax.text(
            dfp.index[-1] + pad * 0.02, y_adj, f"{col} {arrow} {dz:+.2f}",
            va="center", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=0.4),
            clip_on=False, zorder=5,
        )
        ax.plot([dfp.index[-1]], [y0_], marker="o", markersize=3, zorder=4)

    # 軸 & タイトル（JST）
    ax.set_title(
        f"{side.upper()}-score Trails (last {hours}h, top {topn}) — "
        f"{dfp.index[-1].strftime('%Y-%m-%d %H:%M JST')}"
    )
    ax.set_ylabel("Score (z)")
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M", tz=JST))
    plt.setp(ax.get_xticklabels(), rotation=28, ha="right")

    # 上下レンジは±対称に
    lim = max(abs(float(np.nanmin(dfp.values))), abs(float(np.nanmax(dfp.values)))) + 0.1
    ax.set_ylim(-lim, lim)

    fig.tight_layout()
    out_png = out_png or Path(f"score_trails_{side}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[TRAILS] wrote {out_png}")
    return out_png

def send_discord(webhook_url: str, content: str, attachments: List[Path]):
    if not webhook_url:
        print("[WARN] WEBHOOK is empty")
        return
    if requests is None:
        print("[WARN] requests not installed; skip sending")
        return
    files = {}
    payload = {"content": content}
    attach_paths = [p for p in attachments if isinstance(p, Path) and p and p.exists()]
    for i, p in enumerate(attach_paths):
        mime = "image/png" if p.suffix.lower() == ".png" else "text/csv"
        files[f"file{i+1}"] = (p.name, p.read_bytes(), mime)
    try:
        if files:
            r = requests.post(
                webhook_url,
                data={"payload_json": json.dumps(payload, ensure_ascii=False)},
                files=files,
                timeout=30,
            )
        else:
            r = requests.post(webhook_url, json=payload, timeout=30)
        print(f"[DISCORD] status={r.status_code} body={r.text[:200]}")
        r.raise_for_status()
        print(f"[OK] sent to Discord ({len(files)} attachments)")
    except Exception as e:
        print(f"[WARN] Discord webhook failed: {e}")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_largecap_compare.yaml")
    ap.add_argument("--interval-min", type=int, default=60)
    ap.add_argument("--repeat", type=int, default=0, help="0=無限, N=回数")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--webhook-url", default="")
    ap.add_argument("--label", default="LargeCap USD×BTC")

    # 推移グラフオプション（このスクリプト内で描画）
    ap.add_argument("--trails", action="store_true", help="score_trails_*.png を作成（CSVは既存を使用）")
    ap.add_argument("--hours", type=int, default=168, help="推移グラフの時間窓（h）")
    ap.add_argument("--topn", type=int, default=20, help="推移で描く銘柄数（最終値上位）")
    ap.add_argument("--smooth", type=int, default=1, help="移動平均窓（1で無効）")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    out_dir = cfg_path.parent
    compare_py = (Path(__file__).parent / "compare_strength.py").resolve()

    count = 0
    while True:
        stamp_h = now_human()
        stamp_f = now_hm()
        print(f"[RUN] {stamp_h}")

        # 1) 比較スクリプトを1回実行（出力は config の out_* に従う）
        ok = run_py(args.python, compare_py, "--config", cfg_path)
        if not ok:
            print("[WARN] compare_strength.py returned non-zero")

        # 2) 出力名を config から取得し、タイムスタンプ付きに複写
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
        csv_name = cfg.get("out_csv", "largecap_compare.csv")
        png_name = cfg.get("out_png_scatter", "largecap_usd_vs_btc.png")

        src_csv = out_dir / csv_name
        src_png = out_dir / png_name
        dst_csv = dst_png = None

        if src_csv.exists():
            dst_csv = out_dir / f"{src_csv.stem}_{stamp_f}{src_csv.suffix}"
            shutil.copy2(src_csv, dst_csv)
            print("[OK] saved", dst_csv)
        else:
            print("[WARN] CSV not found:", src_csv)

        if src_png.exists():
            dst_png = out_dir / f"{src_png.stem}_{stamp_f}{src_png.suffix}"
            shutil.copy2(src_png, dst_png)
            print("[OK] saved", dst_png)
        else:
            print("[WARN] PNG not found:", src_png)

        # 3) 推移グラフ（JST）を作成（CSVは data/score_trails_*.csv を読む想定）
        trails_pngs: List[Path] = []
        if args.trails:
            p_btc = plot_trails("btc", hours=args.hours, topn=args.topn, smooth=args.smooth,
                                out_png=out_dir / f"score_trails_btc_h{args.hours}_top{args.topn}.png")
            p_usd = plot_trails("usd", hours=args.hours, topn=args.topn, smooth=args.smooth,
                                out_png=out_dir / f"score_trails_usd_h{args.hours}_top{args.topn}.png")
            trails_pngs = [p for p in [p_btc, p_usd] if p is not None]

        # 4) Discord へ送信
        if args.webhook-url if False else args.webhook_url:  # guard for hyphen typo
            summary = summarize(src_csv) if src_csv.exists() else "(no csv)"
            content = f"**{args.label}**  {stamp_h}\n{summary}"
            atts = [p for p in [src_png, src_csv, dst_png, dst_csv] if p] + trails_pngs
            send_discord(args.webhook_url, content, atts)

        count += 1
        if args.repeat and count >= args.repeat:
            break
        try:
            mins = max(1, args.interval_min)
            print(f"[SLEEP] {mins} min (Ctrl+Cで停止)")
            time.sleep(mins * 60)
        except KeyboardInterrupt:
            print("\n[STOP] interrupted by user")
            break

if __name__ == "__main__":
    main()
