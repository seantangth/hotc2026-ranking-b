#!/usr/bin/env bash
# G1 tracker-only 10-clip overfit sanity（D090 G0 → 策略 (b)；A10 單卡，估 <1 小時）。
#
# 本腳本只被「執行」——開機由主 session 負責。啟動前置假設（缺任一即 fail-fast）：
#   1. cloud-init 已掛 /root/rearm_selfkill.sh（見下方 SELFKILL 段）。
#   2. 專案 repo 已放 /home/ubuntu/hsot_repo（與 run_oof405_zero_shot_lambda_v1.sh 同慣例）。
#   3. HF_TOKEN 已 export（官方 facebook/sam3 是 gated repo）。
#
# ── SELFKILL（保險自毀計時器）────────────────────────────────────────────
# 本腳本的保險計時器就是 cloud-init 的 /root/rearm_selfkill.sh backstop——
# 啟動前驗證它存在，**刻意不另掛第二顆 nohup sleep**（q100 慣例）。
# 理由＝鐵律「⏰ 自毀計時器可能有兩套」：08-07 實測 drill.sh 自掛的 4 小時保險
# 在實驗差 3 分鐘完成時把機器砍了——rearm 只延了 cloud-init 那套、沒延腳本自
# 掛那套。單一計時器 ⇒ 延長死線只有一個地方要延。延長前仍應
# `pgrep -af selfkill_core.sh` 盤點（D064(f)：孤兒 sleep 不算數）。
set -euo pipefail

export PYTHONUNBUFFERED=1
export PATH="/home/ubuntu/.local/bin:$PATH"
export RCLONE_CONFIG=/home/ubuntu/rclone.conf

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DEST="$GDRIVE/5_outputs/g1_tracker_overfit_20260829"

die() { echo "FATAL: $*" >&2; exit 1; }

: "${HF_TOKEN:?HF_TOKEN is required (gated facebook/sam3); export it before running}"
sudo test -x /root/rearm_selfkill.sh \
  || die "缺 /root/rearm_selfkill.sh：cloud-init selfkill backstop 未就位（本腳本不自掛計時器，backstop 是唯一保險）"

SRC=/home/ubuntu/hsot_repo          # D093：不可叫 hsot（與套件同名撞 namespace）
WORK=/home/ubuntu/g1_tracker_overfit
RAW=/home/ubuntu/train405_raw_archive
FRAMES=/home/ubuntu/train405_fc
CKPT=/home/ubuntu/ckpt
SAM3ENV=/home/ubuntu/sam3env
PY3="$SAM3ENV/bin/python"
SAM3_SHA=96914d2425f90a64f45ca977c2b5165418099543
SAM3_CKPT_SHA256=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
SYNC_PID=""

test -d "$SRC/3_src" || die "repo 不在 $SRC（開機流程應先放好，同 oof405 腳本慣例）"

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
trap 'echo "死於第 $LINENO 行" >&2' ERR

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

# ── 資料（背景並行）────────────────────────────────────────────────────
# 405 序列假色 zips 全拉（實測 ~4.06 GiB、分鐘級）並重跑完整 prep：
# clips 的 start_position/frame_ids 錨在 oof contract 順序（build_hard_clips.py
# :113-116），**只有 prep_oof405_frames.py 能保證同一對齊**——不自寫選擇性
# 解壓去省這 15 分鐘（D036 的整目錄批量 copy，非執行期逐檔 rclone）。
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
  rclone copyto "$GDRIVE/5_outputs/d079_gate0_20260828/hard_clips_vis_rednir.json" \
    /home/ubuntu/hard_clips_vis_rednir.json
  echo GT_READY
) > "$WORK/logs/gt_download.log" 2>&1 &
GT_PID=$!

(
  set -euo pipefail
  curl -fL -H "Authorization: Bearer $HF_TOKEN" \
    -o "$CKPT/sam3.pt" "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt"
  echo "$SAM3_CKPT_SHA256  $CKPT/sam3.pt" | sha256sum -c -
  echo "official gated: https://huggingface.co/facebook/sam3/resolve/main/sam3.pt (sha256 $SAM3_CKPT_SHA256)" \
    > "$WORK/SAM3_WEIGHT_SOURCE.txt"
  echo SAM3_READY
) > "$WORK/logs/sam3_download.log" 2>&1 &
SAM3_PID=$!

# ── 環境（完全隔離 venv；--system-site-packages 是鐵律級禁項）───────────
[ -x "$SAM3ENV/bin/python" ] || uv venv --python 3.12 "$SAM3ENV"
VIRTUAL_ENV="$SAM3ENV" uv pip install -q torch torchvision --torch-backend=auto
VIRTUAL_ENV="$SAM3ENV" uv pip install -q \
  "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}" \
  "numpy<2" ftfy==6.1.1 regex einops psutil scipy av pycocotools numba \
  python-rapidjson pandas pillow tqdm opencv-python-headless pytest
# D086：setuptools 81+ 移除 pkg_resources，sam3/model_builder.py 需要它。
VIRTUAL_ENV="$SAM3ENV" uv pip install -q "setuptools<81"
"$PY3" -c "import torch,sam3,numpy; assert torch.cuda.is_available(); assert numpy.__version__.startswith('1.'); print(torch.__version__, torch.cuda.get_device_name(0))"

for JOB_PID in $DATA_PIDS "$GT_PID" "$SAM3_PID"; do
  wait "$JOB_PID"
done
grep -q GT_READY "$WORK/logs/gt_download.log"
grep -q SAM3_READY "$WORK/logs/sam3_download.log"

# prep：update 覆蓋後必須恰為 405 unique sequence、contract 167174 行
# （同 oof405 腳本的兩道 assert——這是 clips 對齊的唯一保證）。
"$PY3" "$SRC/3_src/prep/prep_oof405_frames.py" \
  --zip-root "$RAW" \
  --gt /home/ubuntu/2026training.csv \
  --frames-out "$FRAMES" \
  --contracts-out "$WORK/contracts" \
  --hash-zips
test "$(find "$FRAMES" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 405
test "$(($(wc -l < "$WORK/contracts/sample_contract.csv") - 1))" -eq 167174

sha256sum "$CKPT/sam3.pt" > "$WORK/checkpoint_sha256.txt"
uv pip freeze --python "$PY3" > "$WORK/sam3_freeze.txt"
nvidia-smi > "$WORK/nvidia_smi.txt" || true

# 環境 sanity：G1 單元測試（真 torch 環境，24 個全跑；紅了就不燒 GPU 時間）。
echo "$(date -Is) UNITTEST_START" | tee -a "$WORK/timeline.txt"
(cd "$SRC" && "$PY3" -m pytest -q 3_src/test_train_tracker_g1.py) \
  2>&1 | tee "$WORK/logs/pytest_g1.log"

# ── 訓練（產物邊跑邊回傳）──────────────────────────────────────────────
sync_progress &
SYNC_PID=$!
echo "$(date -Is) TRAIN_START" | tee -a "$WORK/timeline.txt"
# rc 2=FAIL、3=ABORT 都是**有效判決**（g1_verdict.json 為準），不視為腳本失敗；
# rc 4=ERROR_INPLACE（撞 in-place autograd 錯誤，verdict json 含 [G1-V1]/[G1-V2]
# mitigation 提示）與其他非零 rc（例外、OOM）是真失敗——需人工套一行修改後重跑。
TRAIN_RC=0
"$PY3" "$SRC/3_src/train_tracker_g1/train_g1.py" \
  --clips-json /home/ubuntu/hard_clips_vis_rednir.json \
  --gt-csv /home/ubuntu/2026training.csv \
  --frames-root "$FRAMES" \
  --sam3-ckpt "$CKPT/sam3.pt" \
  --out-dir "$WORK/run" \
  --steps 400 --n-clips 10 --t 8 --stride 2 --image-size 1008 \
  --lr 1e-4 --seed 42 --device cuda \
  2>&1 | tee "$WORK/logs/train_g1.log" || TRAIN_RC=$?
if [ "$TRAIN_RC" -ne 0 ] && [ "$TRAIN_RC" -ne 2 ] && [ "$TRAIN_RC" -ne 3 ]; then
  die "train_g1.py 異常結束 rc=$TRAIN_RC（非 PASS/FAIL/ABORT 判決）"
fi

test -s "$WORK/run/g1_verdict.json"
test -s "$WORK/run/tracker_g1_final.pt"
test -s "$WORK/run/train_log.jsonl"
echo "$(date -Is) TRAIN_DONE verdict_rc=$TRAIN_RC" | tee -a "$WORK/timeline.txt"
cat "$WORK/run/g1_verdict.json"
