#!/usr/bin/env bash
# ============================================================================
# setup_t1_lambda_v2.sh —(v1 死於 ls|head SIGPIPE+pipefail 靜默死亡,已修) 把現役 Lambda A100（原 T2 vis 訓練機）轉化為
# T1 W1 推論機：65 序列 val_split 重跑 + SAM2 mask 快取（E12 前置）。
#
# 產出回傳 gDrive 5_outputs/t1_rerun_20260805/（seq_csv + masks + submission
# + diagnostics）。完成後啟動 4 小時自毀保險（提前手動 terminate 可省錢）。
# 紀律：D014（官方 rclone）、D019（gDrive 真相源）、D016（產出漸進回傳）。
# 檔名帶版本號（B3 教訓：bash 按位元組邊讀邊執行，不覆蓋執行中腳本）。
# ============================================================================
set -euo pipefail
trap 'echo "🚨 腳本死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DATA=~/t1_data
OUT=~/out_t1
SAMURAI=~/samurai
PY=~/t1env/bin/python

echo "=== [1/6] t1env（繼承 Lambda Stack 系統 torch）==="
if [ ! -d ~/t1env ]; then
  python3 -c "import torch; print('system torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  python3 -m venv --system-site-packages ~/t1env
fi
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'"

echo "=== [2/6] SAMURAI（yangchris11/samurai，E02 同源）==="
if [ ! -d "$SAMURAI" ]; then
  git clone --depth 1 https://github.com/yangchris11/samurai.git "$SAMURAI"
fi
$PY -m pip install -q -e "$SAMURAI/sam2" loguru tqdm 2>&1 | tail -2
$PY -c "import sam2; print('sam2 import OK')"

echo "=== [3/6] SAM2.1-Large 權重（gDrive 快取）==="
mkdir -p ~/ckpt
if [ ! -f ~/ckpt/sam2.1_hiera_large.pt ]; then
  rclone copy "$GDRIVE/4_models/pretrained/sam2.1_hiera_large.pt" ~/ckpt/
fi
sz=$(stat -c%s ~/ckpt/sam2.1_hiera_large.pt)
[ "$sz" -gt 800000000 ] || { echo "🚨 權重 ${sz}B 太小，非 Large"; exit 1; }
echo "ckpt OK ($((sz/1000000)) MB)"

echo "=== [4/6] 65 序列假色（update 修正包優先；攤平任意巢狀）==="
mkdir -p "$DATA" ~/fc_zips
n_done=0
while read -r seq; do
  [ -z "$seq" ] && continue
  mod="${seq%%-*}"; name="${seq#*-}"
  case "$mod" in
    nir)    MODDIR="HSI-NIR-FalseColor";;
    rednir) MODDIR="HSI-RedNIR-FalseColor";;
    vis)    MODDIR="HSI-VIS-FalseColor";;
    *) echo "🚨 未知模態：$seq"; exit 1;;
  esac
  dest="$DATA/$seq"
  if [ -d "$dest" ] && [ "$(find "$dest" -name "*.jpg" | wc -l)" -gt 0 ]; then
    n_done=$((n_done+1)); continue
  fi
  z=~/fc_zips/"$seq".zip
  if [ ! -f "$z" ]; then
    if ! rclone copyto "$GDRIVE/1_data/raw_archive/training/update/$MODDIR/$name.zip" "$z" 2>/dev/null; then
      rclone copyto "$GDRIVE/1_data/raw_archive/training/$MODDIR/$name.zip" "$z"
    fi
  fi
  mkdir -p "$dest"; tmp=$(mktemp -d)
  unzip -q "$z" -d "$tmp"
  find "$tmp" -iname "*.jpg" -exec mv {} "$dest"/ \;
  rm -rf "$tmp"
  n=$(find "$dest" -name "*.jpg" | wc -l)
  [ "$n" -gt 0 ] || { echo "🚨 $seq 解壓後無 jpg"; exit 1; }
  first=$(basename "$(find "$dest" -name "*.jpg" -print -quit)")
  case "${first%.jpg}" in
    ''|*[!0-9]*) echo "🚨 $seq 幀名非純數字（$first）——SAM2 loader 會炸"; exit 1;;
  esac
  n_done=$((n_done+1))
  echo "  [$n_done/65] $seq: $n jpg"
done < ~/val_split_v1.txt
echo "資料就緒：$(ls "$DATA" | wc -l) 序列"

echo "=== [5/6] T1 推論（SAMURAI + SAM2.1-L，mask 快取開啟）==="
$PY ~/track_t1.py --frames-root "$DATA" --seq-list ~/val_split_v1.txt \
  --gt-csv ~/2026training.csv --out-dir "$OUT" --mask-cache \
  --samurai-dir "$SAMURAI" --ckpt ~/ckpt/sam2.1_hiera_large.pt

echo "=== [6/6] 回傳 gDrive ==="
rclone copy "$OUT" "$GDRIVE/5_outputs/t1_rerun_20260805" --transfers 8
echo "✅ 全部完成並回傳 5_outputs/t1_rerun_20260805/"

# 4h 自毀保險（提前手動 terminate 即省錢；驗收有問題時機器還在可 debug）
nohup bash -c '
  sleep 14400
  key=$(cat ~/.lambda_key | tr -d "\n"); [ -z "$key" ] && exit 0
  id=$(curl -s -u "$key:" https://cloud.lambdalabs.com/api/v1/instances | python3 -c "
import json,sys
d=json.load(sys.stdin).get(\"data\",[])
m=[i[\"id\"] for i in d if i.get(\"name\")==\"hsot-t2-dry\"]
print(m[0] if m else \"\")")
  [ -n "$id" ] && curl -s -u "$key:" -X POST \
    https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
    -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$id\"]}"
' > ~/self_destruct.log 2>&1 &
echo "⏰ 4h 自毀保險已啟動（self_destruct.log）"
