#!/usr/bin/env bash
# Fresh current-test75 diagnostic for the hardened rankB_robust profile.
# Infrastructure safety is supplied by Lambda cloud-init selfkill. This script
# syncs every outcome to gDrive, then shortens the remaining selfkill deadline.
set -euo pipefail

export PYTHONUNBUFFERED=1
export PATH="/home/ubuntu/.local/bin:$PATH"
export RCLONE_CONFIG=/home/ubuntu/rclone.conf

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
# 🚨 DEST 必須顯式指定，且**絕對不可**指向 5_outputs/rankb_robust_test75_20260822/。
# 那個目錄裡的 run/full_sam3/submission.csv 是 finalize_submission.py 的 selftest (c)
# 讀取的凍結第三腿 fixture（見該檔 FROZEN_THIRD_LEG_FIXTURE）。本腳本的 finish() 會
# `rclone copy "$WORK" "$DEST"`，一旦 DEST 撞上該路徑就會用新 run 的輸出覆寫 fixture，
# 導致 9/7 主檔每次開工都在 selftest (c) die。09-04 稽核 G09。
DEST="${ROBUST_DEST:?請顯式設定 ROBUST_DEST（例：\$GDRIVE/5_outputs/rankb_robust_onepass_YYYYMMDD）；不可用 rankb_robust_test75_20260822（selftest (c) 的凍結 fixture）}"
case "$DEST" in
  *rankb_robust_test75_20260822*)
    echo "FATAL: ROBUST_DEST 指向 selftest (c) 的凍結 fixture 目錄，拒絕執行（稽核 G09）" >&2
    exit 1;;
esac
REPO=/home/ubuntu/hsot_repo          # D093：不可叫 hsot（與套件同名撞 namespace）
SRC="$REPO/3_src"                    # 本腳本用 $SRC/run_ranking_b.py，SRC 須指 3_src
WORK=/home/ubuntu/rankb_robust_test75
FRAMES=/home/ubuntu/test_fc
CKPT=/home/ubuntu/ckpt
SAMURAI=/home/ubuntu/samurai
T1ENV=/home/ubuntu/t1env
SAM3ENV=/home/ubuntu/sam3env
PY1="$T1ENV/bin/python"
PY3="$SAM3ENV/bin/python"
SAMURAI_SHA=76ba195984892b0d1e3db5d9c9f90bb62175680a
SAM3_SHA=96914d2425f90a64f45ca977c2b5165418099543
SAM3_CKPT_SHA256=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e

# 官方 facebook/sam3 是 gated repo：權杖缺失就在開工前停，不要跑到背景下載才失敗。
: "${HF_TOKEN:?HF_TOKEN is required (gated facebook/sam3); export it before running}"

mkdir -p "$WORK" "$CKPT" "$FRAMES"

# 🚨 GPU 健康閘門（08-31 事故換來）：抽到的 A10 有 64 個未修正 ECC 錯誤
# ＋ Xid 48／Xid 64「All reserved rows for bank are remapped」＝顯存壞且備用列已用盡。
# 症狀是 track_t1.py 跑到一半 SIGABRT（terminate called without an active exception），
# **極易被誤判成自己的程式或參數有問題**——08-31 就誤判了一次，燒掉 40 分鐘與 $0.75。
# ⚠️ Lambda 的 instance id 是「本次租用」的編號，壞卡無法列黑名單，只能每次開機現驗。
ECC=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d " ")
case "$ECC" in
  0|"[N/A]"|"N/A"|"") echo "$(date -Is) GPU_ECC_OK ecc=${ECC:-none}" ;;
  *) echo "FATAL: GPU 有 $ECC 個未修正 ECC 錯誤 ⇒ 壞卡，terminate 換一台（勿在此機除錯）" >&2
     exit 1 ;;
esac

finish() {
  RUN_RC=$?
  trap - EXIT
  set +e
  echo "$(date -Is) finish rc=$RUN_RC" | tee -a "$WORK/finish.txt"
  if command -v rclone >/dev/null 2>&1 && [ -f "$RCLONE_CONFIG" ]; then
    rclone copy "$WORK" "$DEST" --transfers 8
  fi
  sudo /root/rearm_selfkill.sh 5
  exit "$RUN_RC"
}
trap finish EXIT

echo "$(date -Is) SETUP_START"
chmod 600 "$RCLONE_CONFIG"

if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | sudo bash >/dev/null
fi
rclone lsf "$GDRIVE/1_data/packed" >/dev/null

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi

# Download data and weights while the two isolated environments are built.
(
  set -euo pipefail
  rclone copyto "$GDRIVE/1_data/packed/t1test_fc_75.tar" /home/ubuntu/test_fc_75.tar
  tar -xf /home/ubuntu/test_fc_75.tar -C "$FRAMES"
  test "$(find "$FRAMES" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 75
  # 09-06 首次演練實撞：preflight [BLOCK] sample-submission（本腳本從未下載 sample）⇒ rc=2、5 分鐘自毀。
  rclone copyto "$GDRIVE/1_data/raw/sample_submisson.csv" /home/ubuntu/sample_submisson.csv
  test -s /home/ubuntu/sample_submisson.csv
  echo DATA_READY
) > "$WORK/data_setup.log" 2>&1 &
DATA_PID=$!

(
  set -euo pipefail
  rclone copyto "$GDRIVE/4_models/pretrained/sam2.1_hiera_large.pt" "$CKPT/sam2.1_hiera_large.pt"
  test "$(stat -c%s "$CKPT/sam2.1_hiera_large.pt")" -gt 800000000
  echo SAM21_READY
) > "$WORK/sam21_download.log" 2>&1 &
SAM21_PID=$!

(
  set -euo pipefail
  : "${HF_TOKEN:?HF_TOKEN is required for the gated facebook/sam3 weight}"
  if ! echo "$SAM3_CKPT_SHA256  $CKPT/sam3.pt" | sha256sum -c - >/dev/null 2>&1; then
    rm -f "$CKPT/sam3.pt"
    curl -fL -H "Authorization: Bearer $HF_TOKEN" \
      -o "$CKPT/sam3.pt" "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt"
  fi
  echo "$SAM3_CKPT_SHA256  $CKPT/sam3.pt" | sha256sum -c -
  echo "official gated: https://huggingface.co/facebook/sam3/resolve/main/sam3.pt (sha256 $SAM3_CKPT_SHA256)" \
    > "$WORK/SAM3_WEIGHT_SOURCE.txt"
  echo SAM3_READY
) > "$WORK/sam3_download.log" 2>&1 &
SAM3_PID=$!

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

[ -x "$SAM3ENV/bin/python" ] || uv venv --python 3.12 "$SAM3ENV"
VIRTUAL_ENV="$SAM3ENV" uv pip install -q torch torchvision --torch-backend=auto
VIRTUAL_ENV="$SAM3ENV" uv pip install -q \
  "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}" \
  "numpy<2" ftfy==6.1.1 regex einops psutil scipy av pycocotools numba \
  python-rapidjson pandas pillow tqdm opencv-python-headless
VIRTUAL_ENV="$SAM3ENV" uv pip install -q "setuptools<81"
"$PY3" -c "import torch,sam3,numpy; assert torch.cuda.is_available(); assert numpy.__version__.startswith('1.'); print(torch.__version__, torch.cuda.get_device_name(0))"

for JOB_PID in "$DATA_PID" "$SAM21_PID" "$SAM3_PID"; do
  wait "$JOB_PID"
done
grep -q DATA_READY "$WORK/data_setup.log"
grep -q SAM21_READY "$WORK/sam21_download.log"
grep -q SAM3_READY "$WORK/sam3_download.log"

sha256sum "$CKPT/sam2.1_hiera_large.pt" "$CKPT/sam3.pt" > "$WORK/checkpoint_sha256.txt"
git -C "$SAMURAI" rev-parse HEAD > "$WORK/samurai_commit.txt"
uv pip freeze --python "$PY1" > "$WORK/t1_freeze.txt"
uv pip freeze --python "$PY3" > "$WORK/sam3_freeze.txt"

echo "$(date -Is) INFERENCE_START"
"$PY1" "$SRC/run_ranking_b.py" \
  --profile rankB_robust \
  --frames-root "$FRAMES" \
  --sample /home/ubuntu/sample_submisson.csv \
  --work-dir "$WORK/run" \
  --out "$WORK/final.csv" \
  --sam3-python "$PY3" \
  --samurai-python "$PY1" \
  --sam3-ckpt "$CKPT/sam3.pt" \
  --samurai-dir "$SAMURAI" \
  --samurai-ckpt "$CKPT/sam2.1_hiera_large.pt" \
  --sam3-source-revision "$SAM3_SHA" --samurai-source-revision "$SAMURAI_SHA" \
  --execute

test -s "$WORK/final.csv"
echo "$(date -Is) INFERENCE_DONE"

