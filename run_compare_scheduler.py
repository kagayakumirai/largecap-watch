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
        qorder = ["A 強×強（本命）","B 強×弱（USD強・BTC優位）","C 弱×強（BTC重・アルト優勢）","D 弱×弱（様子見）"]
        lines.append("Quadrants: " + " / ".join([f"{k}:{counts.get(k,0)}" for k in qorder]))
        return "\n".join(lines)
    except Exception as e:
        return f"(summary failed: {e})"



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
