#!/usr/bin/env bash
# Resume only the inference stage after environments/data/checkpoints are ready.
set -euo pipefail

export PYTHONUNBUFFERED=1
export RCLONE_CONFIG=/home/ubuntu/rclone.conf

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DEST="$GDRIVE/5_outputs/rankb_robust_test75_20260822"
REPO=/home/ubuntu/hsot_repo          # D093：不可叫 hsot（與套件同名撞 namespace）
SRC="$REPO/3_src"                    # 本腳本用 $SRC/run_ranking_b.py，SRC 須指 3_src
WORK=/home/ubuntu/rankb_robust_test75
FRAMES=/home/ubuntu/test_fc
SAMURAI=/home/ubuntu/samurai
PY1=/home/ubuntu/t1env/bin/python
PY3=/home/ubuntu/sam3env/bin/python

finish() {
  RUN_RC=$?
  trap - EXIT
  set +e
  echo "$(date -Is) inference-only finish rc=$RUN_RC" | tee -a "$WORK/finish.txt"
  rclone copy "$WORK" "$DEST" --transfers 8
  sudo /root/rearm_selfkill.sh 5
  exit "$RUN_RC"
}
trap finish EXIT

test -x "$PY1"
test -x "$PY3"
test -f /home/ubuntu/ckpt/sam3.pt
test -f /home/ubuntu/ckpt/sam2.1_hiera_large.pt
test -f /home/ubuntu/sample_submisson.csv
test "$(find "$FRAMES" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 75

echo "$(date -Is) INFERENCE_ONLY_START"
"$PY1" "$SRC/run_ranking_b.py" \
  --profile rankB_robust \
  --frames-root "$FRAMES" \
  --sample /home/ubuntu/sample_submisson.csv \
  --work-dir "$WORK/run" \
  --out "$WORK/final.csv" \
  --sam3-python "$PY3" \
  --samurai-python "$PY1" \
  --sam3-ckpt /home/ubuntu/ckpt/sam3.pt \
  --samurai-dir "$SAMURAI" \
  --samurai-ckpt /home/ubuntu/ckpt/sam2.1_hiera_large.pt \
  --resume-existing-work \
  --execute

test -s "$WORK/final.csv"
echo "$(date -Is) INFERENCE_ONLY_DONE"

