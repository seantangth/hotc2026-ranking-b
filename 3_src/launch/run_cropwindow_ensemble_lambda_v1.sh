#!/usr/bin/env bash
# D100 crop 窗共識：用兩組「同樣合理但不同」的窗重跑 crop-SAM3，與既有 A 組湊三方 medoid。
#
# 問題（08-31 定位）：交付管線 vs 榜上 best 的 0.0097 落差，**全部**來自 SAM3 腿的 crop 階段
#   （v086 −0.01012、v088 只換 crop 序列 −0.01014 ⇒ 非 crop 序列與第三腿零貢獻），
#   且集中在 5 支（v089 −0.00893 ＝ 88%）。其中 4 支的 **full-frame 追蹤完全正常**
#   （IoU 0.98/0.99/0.99/0.93）⇒ **不是模型變差，是 crop 窗的微小差異讓追蹤發散**。
#
# 兩個修法假說已當場否證，故不再找「原因」：
#   (i) 用「crop 與 full 不一致」偵測崩壞 → 崩掉的 5 支落在 0.016–0.790、正常 26 支 0.022–0.961，
#       **完全重疊**（nir-motorcycle4 一致性 0.022 卻沒崩）⇒ 無可用切點，同 D062 的舊教訓。
#   (ii) 離群幀撐大窗 → 2% 穩健包絡使崩掉的 4 支面積縮小 14.1%、其餘 27 支 11.7%，**沒有差別**。
#
# ⇒ 本次不修機制，改**降低窗的運氣成分**：三組窗各跑一次，逐幀取 medoid。
#   誠實預期：成功率約三成（若三組窗在同一支上都對歪，多數決救不回來）。
#
# 設計紀律：
# - 三組窗都必須是 **9/7 當天算得出來的**（只依賴本次 run 自己的 full 輸出，不用任何歷史檔案）。
# - **不在 test75 上挑窗**——那正是本次診斷出的病（v023 的「好窗」是 08-09 掃參數挑出來的）。
#   三組是「同一原則的不同保守度」，不是擇優。
# - 只重跑 SAM3 腿：問題在它；SAMURAI 腿（每序列重置）已實測比舊版好 +0.00064（v087），沿用。
set -euo pipefail
trap 'echo "死於第 $LINENO 行" >&2' ERR

export PYTHONUNBUFFERED=1
export PATH="/home/ubuntu/.local/bin:$PATH"
export RCLONE_CONFIG=/home/ubuntu/rclone.conf

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DEST="$GDRIVE/5_outputs/cropwindow_ensemble_20260831"
SRC_RUN="$GDRIVE/5_outputs/consensus_2runs_20260830/run_D"   # A 組（現行窗）的 full 輸出來源
REPO=/home/ubuntu/hsot_repo
SRC="$REPO/3_src"
# D093 第 2 條坑（08-31 再撞一次）：`python -m hsot.crop_rerun` 與 track_t1.py 都需要
# 3_src 在 import path 上。交付管線是逐步驟帶 env {"PYTHONPATH": ...}（run_ranking_b.py:262），
# 獨立腳本沒有那層包裝 ⇒ 必須自己 export，否則 ModuleNotFoundError: No module named 'hsot'。
export PYTHONPATH="$SRC"
WORK=/home/ubuntu/cropwin
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
SYNC_PID=""

die() { echo "FATAL: $*" >&2; exit 1; }
: "${HF_TOKEN:?HF_TOKEN is required}"
sudo test -x /root/rearm_selfkill.sh || die "缺 /root/rearm_selfkill.sh"
sudo /root/rearm_selfkill.sh 240
test -d "$SRC" || die "repo 不在 $REPO"

# GPU 健康閘門（08-31 實測換來）：抽到的 A10 出現 64 個未修正 ECC 錯誤
# ＋ Xid 48／Xid 64「All reserved rows for bank are remapped」＝顯存壞且備用列用盡。
# 症狀是 track_t1.py 跑到一半 SIGABRT（terminate called without an active exception），
# 極易被誤判成自己的程式或參數有問題——本次就誤判了一次。開跑前先查，壞卡立刻換機。
ECC=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d " ")
case "$ECC" in
  0|"[N/A]"|"N/A"|"") : ;;
  *) die "GPU 有 $ECC 個未修正 ECC 錯誤 ⇒ 壞卡，terminate 換一台（勿在此機除錯）" ;;
esac

sync_progress() { while true; do rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 \
  --exclude '*.part' --exclude '*.part/**' --exclude 'cropB/**' --exclude 'cropC/**' || true; sleep 60; done; }

finish() {
  RUN_RC=$?; trap - EXIT; set +e
  [ -n "$SYNC_PID" ] && { kill "$SYNC_PID" 2>/dev/null; wait "$SYNC_PID" 2>/dev/null; }
  echo "$(date -Is) finish rc=$RUN_RC" | tee -a "$WORK/finish.txt"
  rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 \
    --exclude '*.part' --exclude '*.part/**' --exclude 'cropB/**' --exclude 'cropC/**'
  SYNC_RC=$?
  [ "$SYNC_RC" -eq 0 ] && { rclone check "$WORK" "$DEST" --one-way \
     --exclude '*.part' --exclude '*.part/**' --exclude 'cropB/**' --exclude 'cropC/**' || SYNC_RC=$?; }
  if [ "$SYNC_RC" -eq 0 ]; then sudo /root/rearm_selfkill.sh 5
  else echo "$(date -Is) REMOTE_VERIFY_FAILED rc=$SYNC_RC" | tee -a "$WORK/finish.txt"; fi
  exit "$RUN_RC"
}
trap finish EXIT

mkdir -p "$WORK/logs" "$CKPT" "$FRAMES"
chmod 600 "$RCLONE_CONFIG"
echo "$(date -Is) SETUP_START" | tee "$WORK/timeline.txt"
command -v rclone >/dev/null 2>&1 || curl -fsSL https://rclone.org/install.sh | sudo bash >/dev/null
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null

# ── 資料／權重／A 組既有輸出（並行）─────────────────────────────────────────
(
  set -euo pipefail
  if [ "$(find "$FRAMES" -mindepth 1 -maxdepth 1 -type d | wc -l)" -ne 75 ]; then
    rclone copyto "$GDRIVE/1_data/packed/t1test_fc_75.tar" /home/ubuntu/test_fc_75.tar
    tar -xf /home/ubuntu/test_fc_75.tar -C "$FRAMES"
  fi
  test "$(find "$FRAMES" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 75
  rclone copyto "$GDRIVE/1_data/raw/sample_submisson.csv" /home/ubuntu/sample_submisson.csv
  echo DATA_READY
) > "$WORK/logs/data.log" 2>&1 & DATA_PID=$!

(
  set -euo pipefail
  if ! echo "$SAM3_CKPT_SHA256  $CKPT/sam3.pt" | sha256sum -c - >/dev/null 2>&1; then
    curl -fL -H "Authorization: Bearer $HF_TOKEN" -o "$CKPT/sam3.pt" \
      "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt"
  fi
  echo "$SAM3_CKPT_SHA256  $CKPT/sam3.pt" | sha256sum -c -
  echo SAM3_READY
) > "$WORK/logs/sam3.log" 2>&1 & SAM3_PID=$!

(
  set -euo pipefail
  rclone copyto "$SRC_RUN/full_sam3/submission.csv" "$WORK/A_full_sam3.csv"
  rclone copyto "$SRC_RUN/full_samurai/submission.csv" "$WORK/A_full_samurai.csv"
  rclone copyto "$SRC_RUN/offline_two_pass/main_merged.csv" "$WORK/A_main_merged.csv"
  rclone copyto "$SRC_RUN/offline_two_pass/source_merged.csv" "$WORK/A_source_merged.csv"
  echo AGROUP_READY
) > "$WORK/logs/agroup.log" 2>&1 & A_PID=$!

[ -x "$SAM3ENV/bin/python" ] || uv venv --python 3.12 "$SAM3ENV"
VIRTUAL_ENV="$SAM3ENV" uv pip install -q torch torchvision --torch-backend=auto
VIRTUAL_ENV="$SAM3ENV" uv pip install -q \
  "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}" \
  "numpy<2" ftfy==6.1.1 regex einops psutil scipy av pycocotools numba \
  python-rapidjson pandas pillow tqdm opencv-python-headless
VIRTUAL_ENV="$SAM3ENV" uv pip install -q "setuptools<81"
"$PY3" -c "import torch,sam3,numpy; assert torch.cuda.is_available()"

for P in "$DATA_PID" "$SAM3_PID" "$A_PID"; do wait "$P"; done
grep -q DATA_READY "$WORK/logs/data.log"; grep -q SAM3_READY "$WORK/logs/sam3.log"
grep -q AGROUP_READY "$WORK/logs/agroup.log"
uv pip freeze --python "$PY3" > "$WORK/sam3_freeze.txt"
nvidia-smi > "$WORK/nvidia_smi.txt" || true
"$PY3" -c "import hsot.crop_rerun, hsot.io" \
  || die "hsot 不可 import（PYTHONPATH=$PYTHONPATH）——08-31 首跑即死在此，故設為前置閘門"
sync_progress & SYNC_PID=$!
echo "$(date -Is) SETUP_OK" | tee -a "$WORK/timeline.txt"

# ── 兩組替代窗（皆只用本次 run 自己的 full 輸出 ⇒ 9/7 可複製）──────────────
#   B：--segments 2 —— 每半段各算窗，軌跡範圍較小 ⇒ 窗更緊、放大更大
#   C：不給 --envelope-extra —— 窗只由主腿自己的軌跡決定（不併入 SAMURAI）
run_group() {   # $1=標籤  $2...=prep 的額外參數
  local TAG="$1"; shift
  echo "$(date -Is) ${TAG}_PREP_START" | tee -a "$WORK/timeline.txt"
  "$PY3" -m hsot.crop_rerun prep \
    --frames-root "$FRAMES" --base-csv "$WORK/A_full_sam3.csv" \
    --area-frac-max 0.55 --out-root "$WORK/crop$TAG" --meta "$WORK/meta_$TAG.json" \
    "$@" 2>&1 | tee "$WORK/logs/prep_$TAG.log"
  "$PY3" -c "
import json,sys
m=json.load(open('$WORK/meta_$TAG.json'))
open('$WORK/seqs_$TAG.txt','w').write('\n'.join(sorted(m)))
print(f'[$TAG] crop 序列 {len(m)} 支')"
  echo "$(date -Is) ${TAG}_TRACK_START" | tee -a "$WORK/timeline.txt"
  "$PY3" "$SRC/track_t1.py" --backend sam3 --frames-root "$WORK/crop$TAG" \
    --out-dir "$WORK/track_$TAG" --sam3-ckpt "$CKPT/sam3.pt" --sam3-eval \
    --source-revision "$SAM3_SHA" \
    --seq-list "$WORK/seqs_$TAG.txt" 2>&1 | tee "$WORK/logs/track_$TAG.log"
  "$PY3" -m hsot.crop_rerun merge --base-csv "$WORK/A_full_sam3.csv" \
    --crop-csv "$WORK/track_$TAG/submission.csv" --meta "$WORK/meta_$TAG.json" \
    --out "$WORK/${TAG}_main_merged.csv" 2>&1 | tee "$WORK/logs/merge_$TAG.log"
  test -s "$WORK/${TAG}_main_merged.csv"
  echo "$(date -Is) ${TAG}_DONE" | tee -a "$WORK/timeline.txt"
}

run_group B --envelope-extra "$WORK/A_full_samurai.csv" --segments 2
run_group C

# ── 三方 medoid（只作用在 SAM3 腿）──────────────────────────────────────────
echo "$(date -Is) MEDOID_START" | tee -a "$WORK/timeline.txt"
"$PY3" "$SRC/run_ensemble_medoid.py" \
  "$WORK/A_main_merged.csv" "$WORK/B_main_merged.csv" "$WORK/C_main_merged.csv" \
  --out "$WORK/medoid_main.csv" --report "$WORK/medoid_report.txt" \
  2>&1 | tee "$WORK/logs/medoid.log"
echo "$(date -Is) ALL_DONE" | tee -a "$WORK/timeline.txt"
exit 0
