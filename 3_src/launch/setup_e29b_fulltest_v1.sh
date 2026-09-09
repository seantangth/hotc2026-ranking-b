#!/usr/bin/env bash
# ============================================================================
# setup_e29b_fulltest_v1.sh — E29 第二段：全量 test pan-sharpen → Kaggle 提交檔
#
# 前置：canary（setup_e29_pansharp_v1.sh）判準達成，且 ~/pull_test_hsi.sh 已把
#      30 支 crop 選中序列的 HSI 拉下來（VIS 14 zip／NIR 14＋RedNIR 2 逐檔目錄）。
#      本腳本沿用同一台機器的 sam3env 與 SAM3 權重。
#
# 【base 選擇】base ＝ **v023（LB 0.69292，門檻 0.55 的 fc crop 版）**，不是 E15。
#   理由：v023 已含全部 30 支的 fc crop 結果 ⇒ (a) 省掉重跑整條 fc 腿的 ~14 分鐘 GPU；
#   (b) **pan 腿若有任何序列缺 HSI 而被跳過，該序列自動落回 v023 的 fc crop 結果**，
#   而不是落回無 crop 的 E15——下檔被鎖在「不差於現行最佳」。
#
# 【座標】pan 腿的影像在 mosaic 解析度 ⇒ 輸出座標需**除回 macro** 才能餵給
#   crop_rerun merge（merge 做的是 x + x1，假設 crop CSV 在假色窗座標系）。
#
# 【D038 事前計算】替換比例 ＝ 30 支 / 11,464 幀 ＝ **42.7% of test**。
#   ⚠️ 這是很高的替換比例 ⇒ canary 的最壞崩幅必須很小才發得出去：
#      下檔 ≈ 0.427 × 最壞崩幅，上檔 ≈ 0.427 × 中位改善；**比 > 3:1 不發**。
#   canary 三支若出現任一支 < −0.10，本腳本產出的 CSV **不得提交**。
#
# 【紀律】D016 每階段即時回傳｜D018 用完必 terminate｜D040 無 GT 只做完整性檢查
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DEST="$GDRIVE/5_outputs/e29_fulltest_20260810"
PY3=~/sam3env/bin/python
OUT=~/e29b; mkdir -p "$OUT"
STAMP=~/e29b_timing.txt; : > "$STAMP"
VARIANT="${VARIANT:-equalized_stretch}"
AFM="${AFM:-0.55}"
die() { echo "🚨 $*"; exit 1; }
mark() { echo "$1 $(date +%s)" >> "$STAMP"; echo "⏱  $1 @ $(date +%H:%M:%S)"; }

mark E29B_START
[ -d ~/test_fc ] || die "假色未解開（先跑 pull_test_hsi.sh）"
[ -d ~/test_hsi ] || die "HSI 未拉取"
[ -x "$PY3" ] || die "sam3env 不存在（本腳本沿用 canary 機器）"

echo "=== [1] 拉 base 與窗源軌跡 ==="
rclone copyto "$GDRIVE/5_outputs/submissions/sub_v023_cropwiden55.csv" ~/base_v023.csv
rclone copyto "$GDRIVE/5_outputs/submissions/sub_v006_e15sam3.csv"     ~/e15_test.csv
rclone copyto "$GDRIVE/5_outputs/submissions/exp003_samurai_large.csv" ~/e02_test.csv
rclone copyto "$GDRIVE/1_data/raw/sample_submisson.csv"                ~/sample.csv
rclone copyto "$GDRIVE/3_src/prep/prep_pansharp_test_v1.py"            ~/prep_pansharp_test_v1.py
rclone copyto "$GDRIVE/3_src/prep/pansharp_v1.py"                      ~/pansharp_v1.py
for f in ~/base_v023.csv ~/e15_test.csv ~/e02_test.csv ~/sample.csv ~/prep_pansharp_test_v1.py; do
  [ -s "$f" ] || die "$f 缺失"
done

echo "=== [2] crop 窗表（門檻 $AFM ＝ D061 定案；窗＝E15∪E02，與 v023 同一條規則）==="
mark WIN_START
PYTHONPATH=~ $PY3 -m hsot.crop_rerun prep --frames-root ~/test_fc \
  --base-csv ~/e15_test.csv --envelope-extra ~/e02_test.csv \
  --area-frac-max "$AFM" --out-root ~/crop_fc --meta ~/crop_meta.json | tail -20 \
  || die "crop prep 失敗"
NSEL=$($PY3 -c "import json;print(len(json.load(open('$HOME/crop_meta.json'))))")
echo "窗表 $NSEL 支"
[ "$NSEL" -ge 25 ] || die "只選中 $NSEL 支（預期 30）——門檻或軌跡有異"
mark WIN_DONE

echo "=== [3] pan-sharpen 全量 test（variant=$VARIANT）==="
mark PREP_START
$PY3 ~/prep_pansharp_test_v1.py --fc-root ~/test_fc \
  --hsi-zip-dir ~/test_hsi/zip --hsi-dir-root ~/test_hsi/dir \
  --meta ~/crop_meta.json --base-csv ~/e15_test.csv \
  --out-root ~/e29_test --variant "$VARIANT" | tail -40 || die "pan-sharpen prep 失敗"
NPAN=$($PY3 -c "import json;print(len(json.load(open('$HOME/e29_test/pansharp_test_meta.json'))))")
echo "pan 腿實際產出 $NPAN / $NSEL 支（缺的會自動落回 v023 的 fc crop）"
mark PREP_DONE
rclone copyto ~/e29_test/pansharp_test_meta.json "$DEST/pansharp_test_meta.json"

echo "=== [4] SAM3 追蹤 pan 腿 ==="
mark TRACK_START
$PY3 ~/track_t1.py --frames-root ~/e29_test --out-dir "$OUT/pan" \
  --backend sam3 --sam3-version sam3 --sam3-ckpt ~/ckpt/sam3.pt || die "pan 腿追蹤失敗"
mark TRACK_DONE
rclone copy "$OUT" "$DEST" --transfers 8

echo "=== [5] 座標除回 macro → merge ==="
$PY3 - <<'PY' || die "座標換算失敗"
import json, os, pandas as pd
H = os.path.expanduser
meta = json.load(open(H("~/e29_test/pansharp_test_meta.json")))
d = pd.read_csv(H("~/e29b/pan/submission.csv"))
q = d["ID"].str.rsplit("_", n=1, expand=True)
d["seq"] = q[0]
for name, m in meta.items():
    k = d["seq"] == name
    for c in ("x", "y", "width", "height"):
        d.loc[k, c] = d.loc[k, c] / m["macro"]
d[["ID", "x", "y", "width", "height"]].to_csv(H("~/e29b/pan_fcscale.csv"), index=False)
print(f"已除回 macro：{len(meta)} 支")
PY
PYTHONPATH=~ $PY3 -m hsot.crop_rerun merge --base-csv ~/base_v023.csv \
  --crop-csv "$OUT/pan_fcscale.csv" --meta ~/crop_meta.json \
  --out "$OUT/sub_e29_pansharp.csv" || die "merge 失敗"

echo "=== [6] 完整性檢查（D040：test 無 GT，只驗完整性）==="
$PY3 - <<'PY' || die "完整性檢查未過"
import os, pandas as pd
H = os.path.expanduser
sub = pd.read_csv(H("~/e29b/sub_e29_pansharp.csv"))
smp = pd.read_csv(H("~/sample.csv"))
base = pd.read_csv(H("~/base_v023.csv"))
errs = []
if set(sub.ID) != set(smp.ID): errs.append("ID 集合與 sample 不符")
if sub.ID.duplicated().any(): errs.append("有重複 ID")
if sub.isna().sum().sum(): errs.append("有 NaN")
if not ((sub.width > 0).all() and (sub.height > 0).all()): errs.append("有非正 w/h")
if (sub.ID.values != smp.ID.values).any(): errs.append("ID 順序與 sample 不同")
ch = (sub.merge(base, on="ID", suffixes=("", "_b"))
      .eval("x!=x_b or y!=y_b or width!=width_b or height!=height_b"))
print(f"列數 {len(sub)}｜與 v023 有差異的幀 {ch.sum()} ({ch.mean():.1%})")
if errs:
    for e in errs: print("🚨", e)
    raise SystemExit(1)
print("✅ 完整性全過，可提交")
PY

rclone copy "$OUT" "$DEST" --transfers 8
rclone copyto "$OUT/sub_e29_pansharp.csv" "$GDRIVE/5_outputs/submissions/sub_e29_pansharp_$(date +%m%d).csv"
mark E29B_DONE
$PY3 - <<'TIME'
import os
H = os.path.expanduser
t = {k: int(v) for k, v in (l.split() for l in open(H("~/e29b_timing.txt")).read().split("\n") if l.strip())}
def span(a, b, label):
    if a in t and b in t:
        s = t[b] - t[a]; print(f"  {label:28s} {s//60:3d} 分 {s%60:02d} 秒")
span("E29B_START", "WIN_DONE", "窗表")
span("PREP_START", "PREP_DONE", "pan-sharpen prep（CPU）")
span("TRACK_START", "TRACK_DONE", "SAM3 追蹤 pan 腿")
span("E29B_START", "E29B_DONE", "★ 總計")
TIME
echo "E29B_DONE — 提交檔 $OUT/sub_e29_pansharp.csv"
echo "⚠️ 用完務必 API terminate（D018）"
