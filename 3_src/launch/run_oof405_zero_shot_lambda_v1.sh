#!/usr/bin/env bash
# train405 frozen prediction cache：SAMURAI(reset) + SAM3 full-frame causal。
# 真正的 grouped cross-fit（corr/K/qhead 等）在 raw cache 完成後另於 CPU 執行。
set -euo pipefail

export PYTHONUNBUFFERED=1
export PATH="/home/ubuntu/.local/bin:$PATH"
export RCLONE_CONFIG=/home/ubuntu/rclone.conf

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DEST="$GDRIVE/5_outputs/oof405_exact_pair_20260827"

# 官方 facebook/sam3 是 gated repo：權杖缺失就在開工前停，不要跑到背景下載才失敗。
: "${HF_TOKEN:?HF_TOKEN is required (gated facebook/sam3); export it before running}"
SRC=/home/ubuntu/hsot_repo          # D093：不可叫 hsot（與套件同名撞 namespace）
WORK=/home/ubuntu/oof405_zero_shot
RAW=/home/ubuntu/train405_raw_archive
FRAMES=/home/ubuntu/train405_fc
CKPT=/home/ubuntu/ckpt
SAMURAI=/home/ubuntu/samurai
T1ENV=/home/ubuntu/t1env
SAM3ENV=/home/ubuntu/sam3env
PY1="$T1ENV/bin/python"
PY3="$SAM3ENV/bin/python"
SAMURAI_SHA=76ba195984892b0d1e3db5d9c9f90bb62175680a
SAM3_SHA=96914d2425f90a64f45ca977c2b5165418099543
SAM3_CKPT_SHA256=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
SYNC_PID=""

sync_progress() {
  while true; do
    rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 \
      --exclude '*.part' --exclude '*.part/**' || true
    sleep 60
  done
}

finish() {
  RUN_RC=$?
  trap - EXIT
  set +e
  if [ -n "$SYNC_PID" ]; then
    kill "$SYNC_PID" 2>/dev/null
    wait "$SYNC_PID" 2>/dev/null
  fi
  echo "$(date -Is) finish rc=$RUN_RC" | tee -a "$WORK/finish.txt"
  SYNC_RC=0
  if command -v rclone >/dev/null 2>&1 && [ -f "$RCLONE_CONFIG" ]; then
    rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 \
      --exclude '*.part' --exclude '*.part/**' || SYNC_RC=$?
    if [ "$SYNC_RC" -eq 0 ]; then
      rclone check "$WORK" "$DEST" --one-way \
        --exclude '*.part' --exclude '*.part/**' || SYNC_RC=$?
    fi
  else
    SYNC_RC=127
  fi
  if [ "$SYNC_RC" -eq 0 ]; then
    sudo /root/rearm_selfkill.sh 5
  else
    echo "$(date -Is) REMOTE_VERIFY_FAILED rc=$SYNC_RC; selfkill 未縮短" | tee -a "$WORK/finish.txt"
    if [ "$RUN_RC" -eq 0 ]; then RUN_RC=3; fi
  fi
  exit "$RUN_RC"
}
trap finish EXIT

mkdir -p "$WORK/logs" "$RAW/training" "$RAW/update" "$FRAMES" "$CKPT"
chmod 600 "$RCLONE_CONFIG"
echo "$(date -Is) SETUP_START" | tee "$WORK/timeline.txt"

if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | sudo bash >/dev/null
fi
rclone lsf "$GDRIVE/1_data/raw_archive/training" >/dev/null
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi

# 405 個 update-first 假色 zip，實算總下載約 4.057 GiB。
for MODDIR in HSI-NIR-FalseColor HSI-RedNIR-FalseColor HSI-VIS-FalseColor; do
  (
    set -euo pipefail
    mkdir -p "$RAW/training/$MODDIR" "$RAW/update/$MODDIR"
    rclone copy "$GDRIVE/1_data/raw_archive/training/$MODDIR" "$RAW/training/$MODDIR" \
      --include '*.zip' --transfers 8 --checkers 16
    rclone copy "$GDRIVE/1_data/raw_archive/training/update/$MODDIR" "$RAW/update/$MODDIR" \
      --include '*.zip' --transfers 8 --checkers 16
    echo "$MODDIR READY"
  ) > "$WORK/logs/data_${MODDIR}.log" 2>&1 &
done
DATA_PIDS=$(jobs -pr)

(
  set -euo pipefail
  rclone copyto "$GDRIVE/1_data/raw/2026training.csv" /home/ubuntu/2026training.csv
  test "$(wc -l < /home/ubuntu/2026training.csv)" -eq 169491
  echo GT_READY
) > "$WORK/logs/gt_download.log" 2>&1 &
GT_PID=$!

(
  set -euo pipefail
  rclone copyto "$GDRIVE/4_models/pretrained/sam2.1_hiera_large.pt" "$CKPT/sam2.1_hiera_large.pt"
  test "$(stat -c%s "$CKPT/sam2.1_hiera_large.pt")" -gt 800000000
  echo SAM21_READY
) > "$WORK/logs/sam21_download.log" 2>&1 &
SAM21_PID=$!

(
  set -euo pipefail
  : "${HF_TOKEN:?HF_TOKEN is required for the gated facebook/sam3 weight}"
  curl -fL -H "Authorization: Bearer $HF_TOKEN" \
    -o "$CKPT/sam3.pt" "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt"
  echo "$SAM3_CKPT_SHA256  $CKPT/sam3.pt" | sha256sum -c -
  echo "official gated: https://huggingface.co/facebook/sam3/resolve/main/sam3.pt (sha256 $SAM3_CKPT_SHA256)" \
    > "$WORK/SAM3_WEIGHT_SOURCE.txt"
  echo SAM3_READY
) > "$WORK/logs/sam3_download.log" 2>&1 &
SAM3_PID=$!

# Idempotent: the runner advertises --resume-existing-work, so a restart after a
# mid-run failure must not die on "virtual environment already exists".  Probe the
# interpreter rather than the directory so a half-built venv is still rebuilt.
[ -x "$T1ENV/bin/python" ] || uv venv --python 3.12 "$T1ENV"
if [ ! -d "$SAMURAI/.git" ]; then
  git clone https://github.com/yangchris11/samurai.git "$SAMURAI"
fi
git -C "$SAMURAI" fetch --depth 1 origin "$SAMURAI_SHA"
git -C "$SAMURAI" checkout --detach "$SAMURAI_SHA"
VIRTUAL_ENV="$T1ENV" uv pip install -q torch torchvision --torch-backend=auto
VIRTUAL_ENV="$T1ENV" uv pip install -q -e "$SAMURAI/sam2" \
  scipy loguru tqdm pandas pillow opencv-python-headless
"$PY1" -c "import torch,sam2,pandas; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

# Idempotent: the runner advertises --resume-existing-work, so a restart after a
# mid-run failure must not die on "virtual environment already exists".  Probe the
# interpreter rather than the directory so a half-built venv is still rebuilt.
[ -x "$SAM3ENV/bin/python" ] || uv venv --python 3.12 "$SAM3ENV"
VIRTUAL_ENV="$SAM3ENV" uv pip install -q torch torchvision --torch-backend=auto
VIRTUAL_ENV="$SAM3ENV" uv pip install -q \
  "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}" \
  "numpy<2" ftfy==6.1.1 regex einops psutil scipy av pycocotools numba \
  python-rapidjson pandas pillow tqdm opencv-python-headless
VIRTUAL_ENV="$SAM3ENV" uv pip install -q "setuptools<81"
"$PY3" -c "import torch,sam3,numpy; assert torch.cuda.is_available(); assert numpy.__version__.startswith('1.'); print(torch.__version__, torch.cuda.get_device_name(0))"

for JOB_PID in $DATA_PIDS "$GT_PID" "$SAM21_PID" "$SAM3_PID"; do
  wait "$JOB_PID"
done
grep -q GT_READY "$WORK/logs/gt_download.log"
grep -q SAM21_READY "$WORK/logs/sam21_download.log"
grep -q SAM3_READY "$WORK/logs/sam3_download.log"

# update 覆蓋後必須恰為 405 unique sequence zip。
"$PY1" "$SRC/3_src/prep/prep_oof405_frames.py" \
  --zip-root "$RAW" \
  --gt /home/ubuntu/2026training.csv \
  --frames-out "$FRAMES" \
  --contracts-out "$WORK/contracts" \
  --hash-zips
test "$(find "$FRAMES" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 405
test "$(($(wc -l < "$WORK/contracts/sample_contract.csv") - 1))" -eq 167174

# 若 gDrive 已有前次 run，contract 必須位元級相同才允許拉回逐序列 artifact resume。
REMOTE_CONTRACT=/home/ubuntu/remote_contract_sha256.json
if rclone copyto "$DEST/contracts/contract_sha256.json" "$REMOTE_CONTRACT" 2>/dev/null; then
  cmp "$REMOTE_CONTRACT" "$WORK/contracts/contract_sha256.json"
  rclone copy "$DEST/run" "$WORK/run" --transfers 8 --checkers 16
elif rclone lsf "$DEST/run" --max-depth 1 >/dev/null 2>&1; then
  echo "remote run 存在但缺 contract hash；拒絕混用" >&2
  exit 2
fi

sha256sum "$CKPT/sam2.1_hiera_large.pt" "$CKPT/sam3.pt" > "$WORK/checkpoint_sha256.txt"
git -C "$SAMURAI" rev-parse HEAD > "$WORK/samurai_commit.txt"
uv pip freeze --python "$PY1" > "$WORK/t1_freeze.txt"
uv pip freeze --python "$PY3" > "$WORK/sam3_freeze.txt"

# 此後每 60 秒同步一次逐序列 CSV/status，意外關機可跨 instance resume。
sync_progress &
SYNC_PID=$!
echo "$(date -Is) INFERENCE_START" | tee -a "$WORK/timeline.txt"
"$PY1" "$SRC/3_src/run_ranking_b.py" \
  --profile rankB_robust_crop \
  --allow-offline-two-pass \
  --frames-root "$FRAMES" \
  --sample "$WORK/contracts/sample_contract.csv" \
  --work-dir "$WORK/run" \
  --out "$WORK/reference_rankb_robust_crop.csv" \
  --sam3-python "$PY3" \
  --samurai-python "$PY1" \
  --sam3-ckpt "$CKPT/sam3.pt" \
  --samurai-dir "$SAMURAI" \
  --samurai-ckpt "$CKPT/sam2.1_hiera_large.pt" \
  --sam3-source-revision "$SAM3_SHA" \
  --samurai-source-revision "$SAMURAI_SHA" \
  --resume-existing-work \
  --execute

test -s "$WORK/run/full_samurai/submission.csv"
test -s "$WORK/run/full_sam3/submission.csv"
test -s "$WORK/run/offline_two_pass/main_merged.csv"
test -s "$WORK/run/offline_two_pass/source_merged.csv"
test -s "$WORK/run/offline_two_pass/crop_meta.json"
test -s "$WORK/run/crop_sam3/diagnostics.json"
test -s "$WORK/run/crop_samurai/diagnostics.json"
test -s "$WORK/reference_rankb_robust_crop.csv"
test "$(($(wc -l < "$WORK/reference_rankb_robust_crop.csv") - 1))" -eq 167174
echo "$(date -Is) INFERENCE_DONE" | tee -a "$WORK/timeline.txt"

# Raw predictions alone are not OOF.  Complete the promised CPU-side grouped
# cross-fit before declaring this run finished; the whole WORK tree is copied to
# DEST by the periodic sync and the EXIT handler, so these reports also land on
# gDrive.  This formal cache certifies the rankB_robust_crop pair -- the profile
# intended for 9/7.  The full-frame legs stay on disk and are cross-fit as a
# second, separately-labelled report so both delivery options have an OOF reading.
#
# Pre-registered selector gate (fixed before any number exists, D040/D061):
#   pooled Delta >= +0.002 AND >= 4/5 folds positive AND worst fold >= -0.003.
echo "$(date -Is) CROSSFIT_START" | tee -a "$WORK/timeline.txt"
"$PY1" "$SRC/3_src/oof405_crossfit.py" \
  --contracts "$WORK/contracts" \
  --pair-profile rankB_robust_crop \
  --main "$WORK/run/offline_two_pass/main_merged.csv" \
  --source "$WORK/run/offline_two_pass/source_merged.csv" \
  --main-diagnostics "$WORK/run/full_sam3/diagnostics.json" \
  --source-diagnostics "$WORK/run/full_samurai/diagnostics.json" \
  --main-crop-diagnostics "$WORK/run/crop_sam3/diagnostics.json" \
  --source-crop-diagnostics "$WORK/run/crop_samurai/diagnostics.json" \
  --crop-meta "$WORK/run/offline_two_pass/crop_meta.json" \
  --out-dir "$WORK/crossfit_crop" \
  --corr none,top-only,both \
  --ks 2,3,5,6,8,24 \
  --qhead none,crossfit-v1
test -s "$WORK/crossfit_crop/report.json"
test -s "$WORK/crossfit_crop/candidate_metrics.csv"
test -s "$WORK/crossfit_crop/fold_metrics.csv"

# Option A (one-pass) still needs its own reading so 9/6 is a comparison, not a guess.
"$PY1" "$SRC/3_src/oof405_crossfit.py" \
  --contracts "$WORK/contracts" \
  --pair-profile rankB_robust \
  --main "$WORK/run/full_sam3/submission.csv" \
  --source "$WORK/run/full_samurai/submission.csv" \
  --out-dir "$WORK/crossfit" \
  --corr none,top-only,both \
  --ks 2,3,5,6,8,24 \
  --qhead none,crossfit-v1
test -s "$WORK/crossfit/report.json"
test -s "$WORK/crossfit/candidate_metrics.csv"
test -s "$WORK/crossfit/fold_metrics.csv"
echo "$(date -Is) CROSSFIT_DONE" | tee -a "$WORK/timeline.txt"
