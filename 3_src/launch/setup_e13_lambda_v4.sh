#!/usr/bin/env bash
# ============================================================================
# setup_e13_lambda_v4.sh — E13 canary(v4 最終版:vot shim 替身,新機 hsot-e13-night)
# 相對 v1-v3 的修正:完全隔離 venv(v2)、torch 釘 2.7+cu126(v3)、
# vot-toolkit 以 ~/vot_shim 15 行替身取代(v4,根治 API 不相容)。
# 上傳前置:rclone.conf、.lambda_key、track_t1.py、canary_e13.txt、val_split_v1.txt、
#           2026training.csv、vot_shim/(scp)
# ============================================================================
set -euo pipefail
trap 'echo "🚨 腳本死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1
GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
PY=~/t13env/bin/python
export PYTHONPATH=~/vot_shim

echo "=== [1/5] 隔離 venv + 依賴(無 vot-toolkit)==="
[ -d ~/t13env ] || python3 -m venv ~/t13env
$PY -m pip install -q --upgrade pip 2>&1 | tail -1
$PY -m pip install -q torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -1
[ -d ~/HiM2SAM ] || git clone --depth 1 https://github.com/LouisFinner/HiM2SAM.git ~/HiM2SAM
$PY -m pip install -q -e ~/HiM2SAM/sam2 2>&1 | tail -1
$PY -m pip install -q loguru tqdm pandas scipy opencv-python-headless matplotlib imageio 2>&1 | tail -1
$PY - <<'PYEOF'
import torch, numpy
from vot.region.raster import calculate_overlaps
from vot.region.shapes import Mask
from vot.region import RegionType
import sam2
assert torch.cuda.is_available(), "CUDA 不可用"
print(f"env OK: torch {torch.__version__} numpy {numpy.__version__} | vot=shim({Mask.__module__})")
PYEOF

echo "=== [2/5] 資料(D036 首次實戰:tar 大包)==="
if [ ! -d ~/t1_data ] || [ "$(ls ~/t1_data | wc -l)" -lt 65 ]; then
  time rclone copy "$GDRIVE/1_data/packed/t1val_fc_65.tar" ~/
  mkdir -p ~/t1_data && time tar xf ~/t1val_fc_65.tar -C ~/t1_data
fi
n_seq=$(ls ~/t1_data | wc -l)
echo "序列數:$n_seq"
[ "$n_seq" -ge 65 ] || { echo "🚨 tar 內容不足 65 序列"; exit 1; }
mkdir -p ~/ckpt
[ -f ~/ckpt/sam2.1_hiera_large.pt ] || rclone copy "$GDRIVE/4_models/pretrained/sam2.1_hiera_large.pt" ~/ckpt/
sz=$(stat -c%s ~/ckpt/sam2.1_hiera_large.pt); [ "$sz" -gt 800000000 ] || { echo "🚨 權重不完整"; exit 1; }

echo "=== [3/5] CoTracker3 預載 ==="
$PY -c "import torch; torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline'); print('cotracker3 OK')"

echo "=== [4/5] E13 canary(25 序列,him2sam/lasot 預設 config)==="
$PY ~/track_t1.py --frames-root ~/t1_data --seq-list ~/canary_e13.txt \
  --gt-csv ~/2026training.csv --out-dir ~/out_e13_canary --mask-cache \
  --samurai-dir ~/HiM2SAM --model-cfg configs/him2sam/lasot/sam2.1_hiera_l.yaml \
  --ckpt ~/ckpt/sam2.1_hiera_large.pt

echo "=== [5/5] 回傳 ==="
rclone copy ~/out_e13_canary "$GDRIVE/5_outputs/e13_canary_20260805" --transfers 8
echo "=== E13 canary done ==="

# 6h 自毀保險(今晚可能續跑全量/test,不立即殺;斷線兜底)
nohup bash -c '
  sleep 21600
  key=$(cat ~/.lambda_key | tr -d "\n"); [ -z "$key" ] && exit 0
  id=$(curl -s -u "$key:" https://cloud.lambdalabs.com/api/v1/instances | python3 -c "
import json,sys
d=json.load(sys.stdin).get(\"data\",[])
m=[i[\"id\"] for i in d if i.get(\"name\")==\"hsot-e13-night\"]
print(m[0] if m else \"\")")
  [ -n "$id" ] && curl -s -u "$key:" -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
    -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$id\"]}"
' > ~/self_destruct6h.log 2>&1 &
echo "⏰ 6h 自毀保險已掛"
