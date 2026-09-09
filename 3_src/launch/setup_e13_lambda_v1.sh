#!/usr/bin/env bash
# ============================================================================
# setup_e13_lambda_v1.sh — E13 HiM2SAM canary(25 序列:Top-15 失分 + 10 穩定)
# 前提:同機已完成 setup_t1_lambda_v2.sh(t1_data 假色、ckpt、2026training.csv、
# track_t1.py、canary_e13.txt 皆就位)。獨立 t13env(兩個 sam2 fork 不同 venv 隔離)。
# 驗收(D032/D033):canary 對照 T1 逐序列 delta;E03 式廣泛退步(改善<退步)即砍。
# ============================================================================
set -euo pipefail
trap 'echo "🚨 腳本死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
PY=~/t13env/bin/python

echo "=== [1/4] t13env + HiM2SAM ==="
[ -d ~/t13env ] || python3 -m venv --system-site-packages ~/t13env
[ -d ~/HiM2SAM ] || git clone --depth 1 https://github.com/LouisFinner/HiM2SAM.git ~/HiM2SAM
$PY -m pip install -q -e ~/HiM2SAM/sam2 loguru tqdm opencv-python-headless scipy matplotlib 2>&1 | tail -2
$PY -c "import sam2; print('HiM2SAM sam2 import OK:', sam2.__file__)"

echo "=== [2/4] 預載 CoTracker3(torch.hub)==="
$PY -c "import torch; m = torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline'); print('cotracker3 OK')"

echo "=== [3/4] E13 canary 推論(him2sam/lasot 預設 config,D033 不調參)==="
$PY ~/track_t1.py --frames-root ~/t1_data --seq-list ~/canary_e13.txt \
  --gt-csv ~/2026training.csv --out-dir ~/out_e13_canary --mask-cache \
  --samurai-dir ~/HiM2SAM --model-cfg configs/him2sam/lasot/sam2.1_hiera_l.yaml \
  --ckpt ~/ckpt/sam2.1_hiera_large.pt

echo "=== [4/4] 回傳 ==="
rclone copy ~/out_e13_canary "$GDRIVE/5_outputs/e13_canary_20260805" --transfers 8
echo "✅ E13 canary 完成並回傳"
