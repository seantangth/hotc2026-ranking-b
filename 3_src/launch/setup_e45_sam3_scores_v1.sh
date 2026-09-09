#!/usr/bin/env bash
# setup_e45_sam3_scores_v1.sh — SAM3 val65 重跑，框與 obj_score 同一輪 propagate。
# DRY=1 支；FULL=val_split_v1 65 支。產出 submission.csv + obj_scores.csv。
# 不用腳本內第二顆 sleep-terminate（cloud-init selfkill 已在）。
set -euo pipefail
trap 'echo "🚨 死於第 $LINENO 行 exit=$?"; echo FAILED > "${STATUS:-$HOME/e45/status.txt}"' ERR
export PYTHONUNBUFFERED=1

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
WORK="${WORK:-$HOME/e45}"
DATA="$WORK/data"
OUT="$WORK/out"
CKPT="$WORK/ckpt"
LOG="$WORK/e45.log"
STATUS="$WORK/status.txt"
MODE="${1:-FULL}"
mkdir -p "$DATA" "$OUT" "$CKPT" "$WORK"
echo RUNNING > "$STATUS"
exec >>"$LOG" 2>&1
echo "=== E45 $(date -u +%Y-%m-%dT%H:%M:%SZ) MODE=$MODE ==="

if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | sudo bash
fi
rclone lsf "$GDRIVE/" >/dev/null

T1="$HOME/hsot_src/3_src/track_t1.py"
[ -f "$T1" ] || T1="$HOME/hsot_src/3_src/3_src/track_t1.py"
[ -f "$T1" ] || T1="$HOME/track_t1.py"
cp "$T1" "$WORK/track_t1.py"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"

# 資料 ∥ 環境
(
  set -euo pipefail
  if [ ! -f "$DATA/.tar_done" ]; then
    rclone copyto "$GDRIVE/1_data/packed/t1val_fc_65.tar" "$WORK/t1val_fc_65.tar"
    tar -xf "$WORK/t1val_fc_65.tar" -C "$DATA"
    # tar 可能自帶一層
    if [ ! -d "$DATA/vis-ant" ] && [ -d "$DATA/t1val_fc_65" ]; then
      mv "$DATA/t1val_fc_65"/* "$DATA/" || true
    fi
    touch "$DATA/.tar_done"
  fi
  rclone copyto "$GDRIVE/1_data/val_split_v1.txt" "$WORK/val_split_v1.txt"
  rclone copyto "$GDRIVE/1_data/raw/2026training.csv" "$WORK/2026training.csv"
  echo DATA_READY
) > "$WORK/data_pull.log" 2>&1 &
DATA_PID=$!

if [ ! -d "$WORK/sam3env" ]; then
  uv venv --python 3.12 "$WORK/sam3env"
fi
PY="$WORK/sam3env/bin/python"
VIRTUAL_ENV="$WORK/sam3env" uv pip install -q torch torchvision --torch-backend=auto
VIRTUAL_ENV="$WORK/sam3env" uv pip install -q "git+https://github.com/facebookresearch/sam3.git" \
  "numpy<2" ftfy==6.1.1 regex einops psutil scipy av \
  pycocotools numba python-rapidjson pandas pillow tqdm opencv-python-headless
VIRTUAL_ENV="$WORK/sam3env" uv pip install -q "setuptools<81"
"$PY" -c "import torch,sam3; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

CKPT_FILE=sam3.pt
if [ ! -f "$CKPT/$CKPT_FILE" ]; then
  if rclone copyto "$GDRIVE/4_models/pretrained/sam3.pt" "$CKPT/$CKPT_FILE" 2>/dev/null; then
    :
  else
    curl -fL -o "$CKPT/$CKPT_FILE" "https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt"
  fi
fi
sz=$(stat -c%s "$CKPT/$CKPT_FILE")
echo "ckpt $sz bytes"
[ "$sz" -gt 3000000000 ]

wait $DATA_PID
nseq=$(find "$DATA" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "val dirs $nseq"
# flatten if tar nested
if [ "$nseq" -lt 10 ]; then
  find "$DATA" -mindepth 2 -maxdepth 2 -type d -name 'vis-*' -o -name 'nir-*' -o -name 'rednir-*' | head
fi

run_one() {
  local list="$1" outd="$2"
  "$PY" "$WORK/track_t1.py" --frames-root "$DATA" --seq-list "$list" \
    --gt-csv "$WORK/2026training.csv" --out-dir "$outd" \
    --backend sam3 --sam3-ckpt "$CKPT/$CKPT_FILE"
}

# DRY
DRY_LIST="$WORK/dry.txt"
grep -m1 '^rednir-' "$WORK/val_split_v1.txt" > "$DRY_LIST" || grep -m1 '^vis-' "$WORK/val_split_v1.txt" > "$DRY_LIST"
echo "DRY $(cat "$DRY_LIST")"
run_one "$DRY_LIST" "$OUT/dry"
test -f "$OUT/dry/obj_scores.csv"
rclone copy "$OUT/dry" "$GDRIVE/5_outputs/e45_sam3_scores_20260819/dry/" --transfers 4
echo "=== DRY 過關 ==="
[ "$MODE" = DRY ] && { echo DRY_DONE > "$STATUS"; exit 0; }

run_one "$WORK/val_split_v1.txt" "$OUT/val"
test -f "$OUT/val/obj_scores.csv"
rclone copy "$OUT/val/submission.csv" "$GDRIVE/5_outputs/e45_sam3_scores_20260819/"
rclone copy "$OUT/val/obj_scores.csv" "$GDRIVE/5_outputs/e45_sam3_scores_20260819/"
rclone copy "$OUT/val/diagnostics.json" "$GDRIVE/5_outputs/e45_sam3_scores_20260819/"
echo "=== E45 FULL 完成 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo DONE > "$STATUS"
