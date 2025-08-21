#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_strength.py を一定間隔で実行し、
- 通常の PNG/CSV を保存＆Discord送信
- 履歴を集計して、スコア推移グラフ(score_trails_*.png)も作成＆送信
"""

import argparse, subprocess, time, datetime as dt, sys, shutil
from pathlib import Path
import yaml, requests, pandas as pd
import json
from zoneinfo import ZoneInfo  # py>=3.9
JST = ZoneInfo("Asia/Tokyo")

def now_hm():    return dt.datetime.now(JST).strftime("%Y%m%d_%H%M")            # ← JST
def now_human(): return dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")
# ==== 先頭付近：import に追加 ====
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd

# ==== ユーティリティ ====
TRAILS_DIR = Path("data")
TRAILS_DIR.mkdir(exist_ok=True)

def _now_utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def persist_trails(side: str, df_now: pd.DataFrame, hours_keep: int = 168):
    """
    side: 'usd' or 'btc'
    df_now: columns=['symbol','score'] を想定（その時点のスコア）
    """
    ts = _now_utc_iso()
    cur = df_now[["symbol", "score"]].copy()
    cur.insert(0, "ts", ts)
    cur.insert(1, "side", side)

    out = TRAILS_DIR / f"score_trails_{side}.csv"
    header = not out.exists()
    cur.to_csv(out, index=False, mode="a", header=header)

    # 古い行を掃除（hours_keepより古いものを落とす）
    try:
        hist = pd.read_csv(out, parse_dates=["ts"])
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_keep+6)  # 余裕6h
        hist = hist[hist["ts"] >= cutoff]
        hist.sort_values(["ts", "symbol"], inplace=True)
        hist.drop_duplicates(["ts", "symbol"], keep="last", inplace=True)
        hist.to_csv(out, index=False)
    except Exception:
        pass
    return out

def load_trails(side: str):
    out = TRAILS_DIR / f"score_trails_{side}.csv"
    if not out.exists():
        return None
    df = pd.read_csv(out, parse_dates=["ts"])
    df.sort_values(["ts", "symbol"], inplace=True)
    return df




# ---------- helpers ----------
def now_hm(): return dt.datetime.now().strftime("%Y%m%d_%H%M")
def now_human(): return dt.datetime.now().strftime("%Y-%m-%d %H:%M")

def run_py(python_exe, script_path, *args):
    cmd = [python_exe, str(script_path), *[str(a) for a in args]]
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout: print(proc.stdout)
    if proc.stderr: print(proc.stderr, file=sys.stderr)
    return proc.returncode == 0

def summarize(csv_path: Path, top_n: int = 5) -> str:
    try:
        df = pd.read_csv(csv_path)
        df_sorted = df.sort_values(["btc_score","usd_score"], ascending=False)
        top = df_sorted.head(top_n)[["symbol","usd_score","btc_score","quadrant"]]
        counts = df["quadrant"].value_counts().to_dict()
        lines = [f"Top{top_n} (btc→usd):"]
        for _, r in top.iterrows():
            lines.append(f"- {r['symbol']}: usd={r['usd_score']:.2f}, btc={r['btc_score']:.2f}, {r['quadrant']}")

            # ==== 最新点をCSVに追記 ====
            # 例: latest_usd, latest_btc を作る（列名を合わせる）
            latest_usd = df_usd.rename(columns={"usd_score":"score"})[["symbol","score"]]
            latest_btc = df_btc.rename(columns={"btc_score":"score"})[["symbol","score"]]
            persist_trails("usd", latest_usd, hours_keep=168)
            persist_trails("btc", latest_btc, hours_keep=168)

        qorder = ["A 強×強（本命）","B 強×弱（USD強・BTC優位）","C 弱×強（BTC重・アルト優勢）","D 弱×弱（様子見）"]
        lines.append("Quadrants: " + " / ".join([f"{k}:{counts.get(k,0)}" for k in qorder]))
        return "\n".join(lines)
    except Exception as e:
        return f"(summary failed: {e})"


# ==== 履歴を読み込み、上位銘柄の Trails を描画 ====
def plot_trails(side: str, topn: int = 20, fname: str = None):
    hist = load_trails(side)
    if hist is None or hist.empty:
        print(f"[TRAILS] no data for {side}")
        return None

    # 最新時点の上位 topn を選ぶ
    last_ts = hist["ts"].max()
    latest = hist[hist["ts"] == last_ts].sort_values("score", ascending=False)
    keep_syms = latest["symbol"].head(topn).tolist()

    dfp = (
        hist[hist["symbol"].isin(keep_syms)]
        .pivot(index="ts", columns="symbol", values="score")
        .sort_index()
    )

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    dfp.plot(ax=ax, legend=True)
    ax.axhline(0, color="C0", linewidth=1)
    ax.set_title(f"{side.upper()}-score Trails (last 168h, top {topn})")
    ax.set_ylabel("Score (z)")
    ax.set_xlabel("Time")
    fig.tight_layout()

    if fname is None:
        fname = f"score_trails_{side}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"[TRAILS] wrote {fname}")
    return fname

# 実行
plot_trails("usd", topn=20, fname="score_trails_usd.png")
plot_trails("btc", topn=20, fname="score_trails_btc.png")




def send_discord(webhook_url: str, content: str, attachments: list[Path]):
    if not webhook_url:
        print("[WARN] WEBHOOK is empty")
        return
    files = {}
    for i, p in enumerate([p for p in attachments if p and p.exists()]):
        mime = "image/png" if p.suffix.lower()==".png" else "text/csv"
        files[f"file{i+1}"] = (p.name, p.read_bytes(), mime)
    payload = {"content": content}

    try:
        if files:
            r = requests.post(
                webhook_url,
                data={"payload_json": json.dumps(payload, ensure_ascii=False)},
                files=files, timeout=30
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
    ap.add_argument("--repeat", type=int, default=0, help="0=無限")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--webhook-url", default="")
    ap.add_argument("--label", default="LargeCap USD×BTC")
    # 推移グラフのオプション
    ap.add_argument("--trails", action="store_true", help="推移グラフも作成して送信")
    ap.add_argument("--hours", type=int, default=24, help="推移グラフの時間窓")
    ap.add_argument("--topn", type=int, default=20, help="推移で描く銘柄数（最終値上位）")
    ap.add_argument("--smooth", type=int, default=3, help="移動平均窓（1で無効）")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    out_dir = cfg_path.parent
    compare_py = (Path(__file__).parent / "compare_strength.py").resolve()
    hist_py    = (Path(__file__).parent / "history_aggregate.py").resolve()
    trails_py  = (Path(__file__).parent / "plot_score_trails.py").resolve()

    count = 0
    while True:
        stamp_human = now_human()
        stamp_file  = now_hm()
        print(f"[RUN] {stamp_human}")

        # 1) 比較スクリプトを1回実行
        ok = run_py(args.python, compare_py, "--config", cfg_path)
        # 出力名はconfigに合わせて読む
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        csv_name = cfg.get("out_csv", "largecap_compare.csv")
        png_name = cfg.get("out_png_scatter", "largecap_usd_vs_btc.png")
        src_csv = out_dir / csv_name
        src_png = out_dir / png_name

        # 2) タイムスタンプ付きに複写
        dst_csv = dst_png = None
        if src_csv.exists():
            dst_csv = out_dir / f"{src_csv.stem}_{stamp_file}{src_csv.suffix}"
            shutil.copy2(src_csv, dst_csv); print("[OK] saved", dst_csv)
        else:
            print("[WARN] CSV not found:", src_csv)
        if src_png.exists():
            dst_png = out_dir / f"{src_png.stem}_{stamp_file}{src_png.suffix}"
            shutil.copy2(src_png, dst_png); print("[OK] saved", dst_png)
        else:
            print("[WARN] PNG not found:", src_png)

        # 3) 推移グラフ生成（任意）
        trails_pngs = []
        if args.trails:
            # 履歴統合
            run_py(args.python, hist_py)
            # BTC/ USD の2枚作成
            run_py(args.python, trails_py, "--metric", "btc", "--hours", args.hours, "--topn", args.topn, "--smooth", args.smooth)
            run_py(args.python, trails_py, "--metric", "usd", "--hours", args.hours, "--topn", args.topn, "--smooth", args.smooth)
            # 生成ファイル名（plot側の規約）
            trails_pngs = [
                out_dir / f"score_trails_btc_h{args.hours}_top{args.topn}.png",
                out_dir / f"score_trails_usd_h{args.hours}_top{args.topn}.png",
            ]
            for p in trails_pngs:
                print(("[OK] trails created" if p.exists() else "[WARN] trails missing"), p)

        # 4) Discordへ送信
        if args.webhook_url:
            summary = summarize(src_csv) if src_csv.exists() else "(no csv)"
            content = f"**{args.label}**  {stamp_human}\n{summary}"
            atts = [src_png, src_csv] + trails_pngs
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
