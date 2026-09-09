#!/usr/bin/env bash
# ============================================================================
# setup_e19_cropzoom_sam3_v1.sh — E19：小目標 crop-zoom × SAM3（test LB 探針）
#
# 依據（08-06 test 無 GT 丟失指紋診斷，E02 vs E15 逐序列）：
#   凍結率中位 E02 0.056 → E15 0.065  ← SAM3 反而更常跟丟
#   小目標(<32px, 43/75 支) 0.083 → 0.110  ← 明顯更差
#   大目標(≥32px, 32 支)    0.042 → 0.037  ← 略好
# ⇒ **SAM3 的 +0.008 來自「追著時框更準」，不是「更少跟丟」**；剩餘上行空間在小目標保持鎖定。
# SAM3 內部 resize 到 1008²，原圖僅 ~409×216 → 裁切後目標有效解析度可再放大數倍。
#
# 與 E14（在 SAM2.1-L 上 −0.001）的差別：(a) 底座換成已驗證 +0.008 的 SAM3；
# (b) E14 的 −0.001 本身落在 ±0.023 雜訊帶內＝**從未真正被量測過**（D040）；
# (c) 窗口改用 E02+E15 兩條軌跡**聯集**——單一軌跡跟丟時會凍結在錯位置，
#     其 envelope 又特別小、還會通過 40% 面積檢查，把真目標切在窗外。
#
# 選序列規則（確定性、無 GT、不認序列名 → Ranking B 合法）：
#   首幀 sqrt(w·h) < 32px 且 聯集窗面積 < 原圖 40%
#
# 紀律：D036（單 tar）、D018（用完必 terminate）、D016（產出漸進回傳）、D040（LB 為裁判）。
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
PY=~/sam3env/bin/python
OUT=~/e19; mkdir -p "$OUT"

echo "=== [0] rclone ==="
command -v rclone >/dev/null || curl -fsSL https://rclone.org/install.sh | sudo bash
rclone lsf "$GDRIVE/" >/dev/null || { echo "🚨 gDrive 失敗"; exit 1; }

echo "=== [1] 背景拉 test 假色單 tar（已含 init_rect，08-06 修正版）==="
(
  set -euo pipefail
  mkdir -p ~/test_fc
  rclone copyto "$GDRIVE/1_data/packed/t1test_fc_75.tar" ~/t1test.tar
  tar -xf ~/t1test.tar -C ~/test_fc
  d=$(find ~/test_fc -mindepth 1 -maxdepth 1 -type d | wc -l)
  n=$(find ~/test_fc -name init_rect.txt | wc -l)
  [ "$d" -eq 75 ] && [ "$n" -eq 75 ] || { echo "🚨 序列 $d / init_rect $n（應各 75）"; exit 1; }
  echo "TEST_FC_READY"
) > ~/fc.log 2>&1 &
FC_PID=$!

echo "=== [2] sam3env + SAM3 權重（並行）==="
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ~/sam3env ] || uv venv --python 3.12 ~/sam3env
VIRTUAL_ENV=~/sam3env uv pip install -q torch torchvision --torch-backend=auto
mkdir -p ~/ckpt
(
  set -euo pipefail
  [ -f ~/ckpt/sam3.pt ] || curl -fL -o ~/ckpt/sam3.pt "https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt"
  sz=$(stat -c%s ~/ckpt/sam3.pt); [ "$sz" -gt 3400000000 ] || { echo "🚨 權重 $sz 太小"; exit 1; }
  echo CKPT_READY
) > ~/ckpt.log 2>&1 &
CKPT_PID=$!

# SAM3 依賴（08-06 實測補齊；setuptools 必須最後降級）
VIRTUAL_ENV=~/sam3env uv pip install -q "git+https://github.com/facebookresearch/sam3.git" \
  "numpy<2" ftfy==6.1.1 regex einops psutil scipy av \
  pycocotools numba python-rapidjson pandas pillow tqdm opencv-python-headless
VIRTUAL_ENV=~/sam3env uv pip install -q "setuptools<81"
$PY -c "import sam3, numpy; assert numpy.__version__.startswith('1.'); print('sam3 OK')" \
  2>&1 | grep -viE 'warning|deprecat|^ +import'

rclone copyto "$GDRIVE/3_src/track_t1.py" ~/track_t1.py
rclone copy "$GDRIVE/3_src/hsot/" ~/hsot/ --include "*.py"
rclone copyto "$GDRIVE/5_outputs/submissions/exp003_samurai_large.csv" ~/e02_test.csv
rclone copyto "$GDRIVE/5_outputs/submissions/sub_v006_e15sam3.csv" ~/e15_test.csv

wait $CKPT_PID; grep -q CKPT_READY ~/ckpt.log || { echo "🚨 權重失敗"; exit 1; }
wait $FC_PID;  grep -q TEST_FC_READY ~/fc.log || { echo "🚨 test 假色失敗"; exit 1; }
echo "✅ 環境 + 權重 + 資料就緒"

echo "=== [3] crop prep（窗口 = E15 ∪ E02 軌跡聯集）==="
PYTHONPATH=~ $PY -m hsot.crop_rerun prep --frames-root ~/test_fc \
  --base-csv ~/e15_test.csv --envelope-extra ~/e02_test.csv \
  --out-root ~/crop_fc --meta ~/crop_meta.json | tail -30
NSEL=$($PY -c "import json;print(len(json.load(open('$HOME/crop_meta.json'))))")
echo "選中 $NSEL 支"
[ "$NSEL" -ge 5 ] || { echo "🚨 只選中 $NSEL 支，不值得跑"; exit 1; }
$PY -c "
import json,os
m=json.load(open(os.path.expanduser('~/crop_meta.json')))
open(os.path.expanduser('~/crop_seqs.txt'),'w').write('\n'.join(sorted(m))+'\n')"

echo "=== [4] SAM3 追蹤裁切後序列 ==="
$PY ~/track_t1.py --frames-root ~/crop_fc --seq-list ~/crop_seqs.txt \
  --out-dir "$OUT/crop" --backend sam3 --sam3-version sam3 --sam3-ckpt ~/ckpt/sam3.pt

echo "=== [5] merge 回原圖座標（未選序列保留 E15）==="
PYTHONPATH=~ $PY -m hsot.crop_rerun merge --base-csv ~/e15_test.csv \
  --crop-csv "$OUT/crop/submission.csv" --meta ~/crop_meta.json --out "$OUT/e19_full.csv"
$PY - <<'CHK'
import os, pandas as pd
H = os.path.expanduser
b = pd.read_csv(H("~/e15_test.csv")); n = pd.read_csv(H("~/e19/e19_full.csv"))
assert len(n) == len(b) == 26860, f"列數 {len(n)}"
assert (n.ID.values == b.ID.values).all(), "ID 順序不符"
assert n.isna().sum().sum() == 0 and (n.iloc[:,3] > 0).all() and (n.iloc[:,4] > 0).all()
ch = (n.iloc[:,1:].values != b.iloc[:,1:].values).any(1)
print(f"✅ 驗證過：變動 {ch.sum()} 幀 ({ch.mean():.1%})，其餘沿用 E15")
CHK
rclone copy "$OUT" "$GDRIVE/5_outputs/e19_cropzoom_20260806" --transfers 8
echo "E19_DONE"

nohup bash -c '
  sleep 7200
  key=$(cat ~/.lambda_key | tr -d "\n"); [ -z "$key" ] && exit 0
  id=$(curl -s -u "$key:" https://cloud.lambda.ai/api/v1/instances | python3 -c "
import json,sys
d=json.load(sys.stdin).get(\"data\",[])
m=[i[\"id\"] for i in d if i.get(\"name\")==\"hsot-e19\"]
print(m[0] if m else \"\")")
  [ -n "$id" ] && curl -s -u "$key:" -X POST \
    https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$id\"]}"
' > ~/self_destruct.log 2>&1 &
