#!/usr/bin/env bash
# ============================================================================
# setup_e27_cropwiden_v1.sh — crop 面積門檻 0.40 → 0.60（**只新增序列，不動已驗證的 21 支**）
#
# 為何重開（時序證據，非翻案癖）：
#   D043「放寬門檻判死」鑄於 08-06，依據＝`crop_rerun.py` 註解記載的 sweep
#   「k=1 門檻 0.70 → 35 支／放大中位 1.68 ← 放寬納入的都是大窗，把品質拉低」
#   ＝**純解析度框架**。而 D046（08-07 收盤）以 v013/v014 完整 2×2 的 LB 實測
#   **推翻解析度說**、確立「crop 的價值在移除干擾物」⇒ 否決寫在其前提被推翻之前，
#   且 D046 自己的結論「主線改向抑制干擾物／縮小有效視野」從未被套回被剔除組。
#   ⚠️ 當年 sweep 測過 0.40 與 0.70，**從未測過 0.60**。
#
# 08-09 零成本幾何前測（hsot/crop_gap_probe.py）：
#   test 被剔除的 22 支中，**11 支的窗僅 40–60%**（仍能移除 40%+ 的干擾物場）
#   ＝「中等窗」而非當年說的「大窗」，合計 4,244 幀 ＝ **15.8% 的 test 幀**。
#
# 與 v009/E21（分段窗，LB −0.0139）的關鍵差異：
#   E21 敗因是「把已驗證 +0.0175 的 21 支推翻重做」；本實驗**只新增 11 支、
#   v008 的 21 支逐幀原封不動** ⇒ 不是同一類風險。
#
# 事前判準（D038 不對稱比，**寫死於執行前**）：
#   替換比例 15.8%。上檔估 +0.005（現行組 26% 幀換 +0.0175，新增組放大倍率較低故折半）；
#   下檔估 −0.010（保守）⇒ 不對稱比 ≈ 2:1 < 3:1 紅線 ⇒ **可發**。
#   發前必過無 GT 災難檢查：merge 後 vs v008 的差異**只能出現在這 11 支**，
#   其餘 64 支必須逐幀位移 0.00px（否則管線有 bug，停工不發）。
#
# 【紀律】D036 單 tar｜D018 保險自毀＋用完 terminate｜D016 每階段完成立即 rclone
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
GOUT="$GDRIVE/5_outputs/e27_cropwiden_20260809"
PY3=~/sam3env/bin/python
INSTANCE_NAME="${INSTANCE_NAME:-hsot-e27}"
STAMP=~/e27_timing.txt; : > "$STAMP"
die() { echo "🚨 $*"; rclone copyto "$STAMP" "$GOUT/timing_DIED.txt" 2>/dev/null; exit 1; }
mark() { echo "$1 $(date +%s)" >> "$STAMP"; echo "⏱  $1 @ $(date +%H:%M:%S)"; }
mark START

# 幾何前測算出的新增 11 支（窗 40–60%）
NEW_SEQS="rednir-rccar2 nir-herbs6 nir-herbs9 vis-walker2 nir-motorcycle4 nir-pingpong2 vis-truck5 vis-herbs7 nir-jelly3 vis-herbs9 nir-herbs7"

nohup bash -c "
  sleep 5400
  key=\$(cat ~/.lambda_key | tr -d '\n'); [ -z \"\$key\" ] && exit 0
  id=\$(curl -s -u \"\$key:\" https://cloud.lambda.ai/api/v1/instances | python3 -c \"
import json,sys
d=json.load(sys.stdin).get('data',[])
m=[i['id'] for i in d if i.get('name')=='$INSTANCE_NAME']
print(m[0] if m else '')\")
  [ -n \"\$id\" ] && curl -s -u \"\$key:\" -X POST \
    https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H 'Content-Type: application/json' -d \"{\\\"instance_ids\\\":[\\\"\$id\\\"]}\"
" > ~/insurance_destruct.log 2>&1 &
echo "⏰ 90 分鐘保險自毀已掛（$INSTANCE_NAME）"

# --- [1] 資料 ‖ 環境 並行 ----------------------------------------------------
mark ENV_DATA_START
(
  rclone copy "$GDRIVE/1_data/packed/t1test_fc_75.tar" ~/ && tar -xf ~/t1test_fc_75.tar -C ~/
  rclone copyto "$GDRIVE/5_outputs/submissions/exp003_samurai_large.csv" ~/e02_test.csv
  rclone copyto "$GDRIVE/5_outputs/submissions/sub_v006_e15sam3.csv"     ~/e15_test.csv
  rclone copyto "$GDRIVE/5_outputs/submissions/sub_v008_e19cropzoom.csv" ~/v008.csv
  rclone copyto "$GDRIVE/1_data/raw/sample_submisson.csv" ~/sample.csv || true
  echo DATA_READY
) > ~/data.log 2>&1 &
DATA_PID=$!

[ -d ~/sam3env ] || uv venv --python 3.12 ~/sam3env
if [ ! -f ~/sam3env/.deps_done ]; then
  VIRTUAL_ENV=~/sam3env uv pip install -q torch torchvision --torch-backend=auto || die "torch 安裝失敗"
  VIRTUAL_ENV=~/sam3env uv pip install -q "git+https://github.com/facebookresearch/sam3.git@96914d2425f90a64f45ca977c2b5165418099543" \
    "numpy<2" ftfy==6.1.1 regex einops psutil scipy av \
    pycocotools numba python-rapidjson pandas pillow tqdm opencv-python-headless || die "sam3 安裝失敗"
  VIRTUAL_ENV=~/sam3env uv pip install -q "setuptools<81" || die "setuptools 降級失敗"
  touch ~/sam3env/.deps_done || die "deps_done 標記失敗"
fi
$PY3 -c "
import sam3, numpy, pkg_resources, torch
assert torch.cuda.is_available(), 'CUDA 不可用'
assert numpy.__version__.startswith('1.'), f'numpy {numpy.__version__} 應 <2'
print(f'sam3env OK: torch {torch.__version__} numpy {numpy.__version__}')
" || die "sam3env 驗證失敗"

[ -f ~/sam3.pt ] || $PY3 - <<'EOF' || die "權重下載失敗"
from huggingface_hub import hf_hub_download
import os, shutil
p = hf_hub_download("1038lab/sam3", "sam3.pt")
shutil.copy(p, os.path.expanduser("~/sam3.pt"))
print("ckpt OK")
EOF
[ -f ~/sam3.pt ] || die "sam3.pt 不存在"

rclone copyto "$GDRIVE/3_src/track_t1.py" ~/track_t1.py
mkdir -p ~/hsot && rclone copy "$GDRIVE/3_src/hsot/" ~/hsot/ --include "*.py"
touch ~/hsot/__init__.py
grep -q "area_frac_max" ~/hsot/crop_rerun.py || die "crop_rerun.py 非 E27 版（缺門檻參數）"
wait $DATA_PID; grep -q DATA_READY ~/data.log || die "資料就緒失敗（見 ~/data.log）"
FR=$(find ~ -maxdepth 3 -type d -name "nir-bee2" -print -quit | xargs -r dirname)
[ -d "$FR" ] || die "test 假色目錄結構異常"
echo "frames-root = $FR"
mark ENV_DATA_DONE

# --- [2] crop prep：門檻 0.60、**只做新增 11 支** ----------------------------
mark PREP_START
$PY3 -m hsot.crop_rerun prep --frames-root "$FR" --base-csv ~/e02_test.csv \
  --envelope-extra ~/e15_test.csv --out-root ~/crop60 --meta ~/win60.json \
  --area-frac-max 0.60 --only-seqs $NEW_SEQS > ~/prep.log 2>&1 || die "prep 失敗（見 ~/prep.log）"
tail -14 ~/prep.log
NSEL=$($PY3 -c "import json;print(len(json.load(open('$HOME/win60.json'))))")
[ "$NSEL" -ge 8 ] || die "只選中 $NSEL 支（預期 11）——幾何與本機前測不符，停工"
rclone copyto ~/win60.json "$GOUT/win60.json"; rclone copyto ~/prep.log "$GOUT/prep.log"
mark PREP_DONE

# --- [3] 跑這 11 支 -----------------------------------------------------------
mark TRACK_START
ls ~/crop60 > ~/e27_list.txt
$PY3 ~/track_t1.py --frames-root ~/crop60 --seq-list ~/e27_list.txt \
  --out-dir ~/out_e27 --backend sam3 --sam3-ckpt ~/sam3.pt --sam3-eval \
  > ~/track.log 2>&1 || die "track 失敗（見 ~/track.log）"
tail -8 ~/track.log
rclone copy ~/out_e27/ "$GOUT/out_e27/"; rclone copyto ~/track.log "$GOUT/track.log"
mark TRACK_DONE

# --- [4] merge 進 v008 + 無 GT 災難檢查 --------------------------------------
mark MERGE_START
$PY3 -m hsot.crop_rerun merge --base-csv ~/v008.csv --crop-csv ~/out_e27/submission.csv \
  --meta ~/win60.json --out ~/sub_v020_cropwiden60.csv > ~/merge.log 2>&1 || die "merge 失敗"
tail -6 ~/merge.log

$PY3 - > ~/e27_check.txt 2>&1 <<'PYEOF'
import json, os
import numpy as np, pandas as pd
H = os.path.expanduser("~")
NEW = set(json.load(open(f"{H}/win60.json")).keys())
a = pd.read_csv(f"{H}/v008.csv"); b = pd.read_csv(f"{H}/sub_v020_cropwiden60.csv")
assert list(a.columns) == list(b.columns), "欄位不符"
assert len(a) == len(b), f"列數不符 {len(a)} vs {len(b)}"
m = a.merge(b, on="ID", suffixes=("_a", "_b"))
assert len(m) == len(a), "ID 集合不符"
q = m["ID"].str.rsplit("_", n=1, expand=True); m["seq"] = q[0]
d = np.hypot(m["x_a"] + m["width_a"]/2 - (m["x_b"] + m["width_b"]/2),
             m["y_a"] + m["height_a"]/2 - (m["y_b"] + m["height_b"]/2))
m["d"] = d
changed = sorted(m.loc[m["d"] > 1e-9, "seq"].unique())
print(f"預期變動序列（新增 crop）: {len(NEW)} 支")
print(f"實際變動序列: {len(changed)} 支")
unexpected = [s for s in changed if s not in NEW]
print(f"🚨 非預期變動: {unexpected}" if unexpected else "✅ 變動僅限新增序列，其餘 64 支逐幀 0.00px")
for s in changed:
    g = m[m["seq"] == s]
    print(f"  {s:22s} n={len(g):4d}  位移 中位 {g['d'].median():7.2f}px  最大 {g['d'].max():8.2f}px")
frac = float((m["d"] > 1e-9).mean())
print(f"\n變動幀佔比 {frac:.1%}（本機幾何前測預期 15.8%）")
bad = int(((b["width"] <= 0) | (b["height"] <= 0)).sum()) + int(b.isna().any(axis=1).sum())
print(f"完整性：{len(b)} 列｜非正 w/h 或 NaN = {bad} {'✅' if bad == 0 else '🚨'}")
ok = (not unexpected) and bad == 0 and len(b) == len(a)
print(f"\n▶ {'✅ 災難檢查通過 → 可發 LB' if ok else '🚨 檢查未過 → 不發'}")
PYEOF
cat ~/e27_check.txt
rclone copyto ~/e27_check.txt "$GOUT/check.txt"
rclone copyto ~/sub_v020_cropwiden60.csv "$GDRIVE/5_outputs/submissions/sub_v020_cropwiden60.csv"
rclone copyto ~/sub_v020_cropwiden60.csv "$GOUT/sub_v020_cropwiden60.csv"
mark MERGE_DONE
rclone copyto "$STAMP" "$GOUT/timing.txt"

echo "===================== E27 完成 ====================="
cat "$STAMP"
echo "⚠️ 記得 terminate：$INSTANCE_NAME"
