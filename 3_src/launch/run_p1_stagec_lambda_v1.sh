#!/usr/bin/env bash
# P1 Stage C 獨立執行（2026-09-05 晚間，Sean 21:54 授權「趕緊做模型層的改動」）：
#   merged_p1_A.pt（D105 arm A，只訓 sam_mask_decoder 4.2M）跑 test75 rankB_deliver_v090 全鏈、只換 --sam3-ckpt。
#
# 與 run_deliver_v078_drill_lambda_v1.sh 的差別【只有三處，其餘逐字沿用】：
#   1. 多拉 merged_p1_A.pt（3.45GB，gDrive 5_outputs/p1_precision_train_20260902/run/），驗位元數
#   2. run_ranking_b 的 --sam3-ckpt 指向它（full／三窗 crop／第三腿全部換；SAMURAI 腿不動）
#   3. 階段 3 的比對對象是 sub_v090：差異＝模型層改動的足跡，不是「應 100% 相同」；不 die
# 事前判準見 0_README/HSOT_EXPERIMENT_LOG.md「09-05 晚間 P1 Stage C」節（三發：全鏈／v090⊕P1／v087⊕P1）。
# DEST 是獨立資料夾，不寫入任何 selftest fixture（G09）。跳過 pytest（dry-run BLOCK=0 ＋ selftest 四段已足）。
#
# ⏰ 自毀：唯一計時器＝cloud-init /root/rearm_selfkill.sh（launch 帶 90 分）。開工 rearm：
#   A100 300 分（v090 全鏈 A10 實測 4h36m；SAM3 推論 A100 約 2.5–2.8×，估 GPU 段 ~100 分＋CPU/環境 ~40 分＋緩衝）
#   A10  420 分。finish() 同步驗證通過後縮回 5 分。
set -euo pipefail
trap 'echo "死於第 $LINENO 行" >&2' ERR

export PYTHONUNBUFFERED=1
export PATH="/home/ubuntu/.local/bin:$PATH"
export RCLONE_CONFIG=/home/ubuntu/rclone.conf

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
RUN_NAME="${RUN_NAME:-p1_stagec_test75_20260905}"
DEST="$GDRIVE/5_outputs/$RUN_NAME"
P1_SRC="$GDRIVE/5_outputs/p1_precision_train_20260902/run/merged_p1_A.pt"
P1_BYTES=3450165338
REPO=/home/ubuntu/hsot_repo         # D093：不可叫 hsot
SRC="$REPO/3_src"
WORK="/home/ubuntu/$RUN_NAME"
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
SAMPLE=/home/ubuntu/sample_submisson.csv
SELFTEST_SAMPLE=/home/ubuntu/selftest_sample_test75.csv
PINNED_ENV="${PINNED_ENV:-1}"
PROFILE=rankB_deliver_v090
REF_SUB=sub_v090_cropwin_medoid.csv
SYNC_PID=""

die() { echo "FATAL: $*" >&2; exit 1; }

: "${HF_TOKEN:?HF_TOKEN is required (gated facebook/sam3); export it before running}"
sudo test -x /root/rearm_selfkill.sh \
  || die "缺 /root/rearm_selfkill.sh：cloud-init selfkill backstop 未就位（本腳本不自掛計時器）"
test -d "$SRC" || die "repo 不在 $REPO（開機流程須把 3_src 放這裡）"
export PYTHONPATH="$SRC"            # D093／08-31：獨立腳本必須自己帶 PYTHONPATH

# 🚨 GPU 健康閘門（08-31 壞卡事故）＋ 型號記錄（G20：位元級判準只對 A10 成立）
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
ECC=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d " ")
case "$ECC" in
  0|"[N/A]"|"N/A]"|"N/A"|"") echo "$(date -Is) GPU_ECC_OK gpu=$GPU_NAME ecc=${ECC:-none}" ;;
  *) echo "FATAL: GPU 有 $ECC 個未修正 ECC 錯誤 ⇒ 壞卡，terminate 換一台（勿在此機除錯）" >&2; exit 1 ;;
esac
if [[ "$GPU_NAME" == *A100* || "$GPU_NAME" == *H100* ]]; then SELFKILL_MIN="${REARM_MIN:-300}"; else SELFKILL_MIN="${REARM_MIN:-420}"; fi
sudo /root/rearm_selfkill.sh "$SELFKILL_MIN"
echo "profile=$PROFILE ckpt=merged_p1_A.pt gpu=$GPU_NAME selfkill=$SELFKILL_MIN"

SYNC_SKIP=(--exclude 'run/offline_two_pass/frames_*/**')
sync_progress() {
  while true; do
    rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 \
      --exclude '*.part' --exclude '*.part/**' "${SYNC_SKIP[@]}" || true
    sleep 60
  done
}
finish() {
  RUN_RC=$?
  trap - EXIT
  set +e
  [ -n "$SYNC_PID" ] && { kill "$SYNC_PID" 2>/dev/null; wait "$SYNC_PID" 2>/dev/null; }
  echo "$(date -Is) finish rc=$RUN_RC" | tee -a "$WORK/finish.txt"
  SYNC_RC=0
  if command -v rclone >/dev/null 2>&1 && [ -f "$RCLONE_CONFIG" ]; then
    rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 \
      --exclude '*.part' --exclude '*.part/**' "${SYNC_SKIP[@]}" || SYNC_RC=$?
    [ "$SYNC_RC" -eq 0 ] && { rclone check "$WORK" "$DEST" --one-way \
      --exclude '*.part' --exclude '*.part/**' "${SYNC_SKIP[@]}" || SYNC_RC=$?; }
  else
    SYNC_RC=127
  fi
  if [ "$SYNC_RC" -eq 0 ]; then
    sudo /root/rearm_selfkill.sh 5
  else
    echo "$(date -Is) REMOTE_VERIFY_FAILED rc=$SYNC_RC；selfkill 未縮短" | tee -a "$WORK/finish.txt"
    [ "$RUN_RC" -eq 0 ] && RUN_RC=3
  fi
  exit "$RUN_RC"
}
trap finish EXIT

mkdir -p "$WORK/logs" "$CKPT" "$FRAMES"
chmod 600 "$RCLONE_CONFIG"
echo "$(date -Is) SETUP_START gpu=$GPU_NAME" | tee "$WORK/timeline.txt"
echo "$GPU_NAME" > "$WORK/gpu_name.txt"

command -v rclone >/dev/null 2>&1 || curl -fsSL https://rclone.org/install.sh | sudo bash >/dev/null
rclone lsf "$GDRIVE/1_data/packed" >/dev/null
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null

# ── 資料／權重（背景並行；D036 單一 tar）──────────────────────────────────
(
  set -euo pipefail
  rclone copyto "$GDRIVE/1_data/packed/t1test_fc_75.tar" /home/ubuntu/test_fc_75.tar
  tar -xf /home/ubuntu/test_fc_75.tar -C "$FRAMES"
  test "$(find "$FRAMES" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 75
  rclone copyto "$GDRIVE/1_data/raw/sample_submisson.csv" "$SAMPLE"
  test -s "$SAMPLE"
  rclone copyto "$GDRIVE/1_data/raw/sample_submisson.csv" "$SELFTEST_SAMPLE"
  test -s "$SELFTEST_SAMPLE"
  echo DATA_READY
) > "$WORK/logs/data_setup.log" 2>&1 &
DATA_PID=$!

(
  set -euo pipefail
  rclone copyto "$GDRIVE/4_models/pretrained/sam2.1_hiera_large.pt" "$CKPT/sam2.1_hiera_large.pt"
  test "$(stat -c%s "$CKPT/sam2.1_hiera_large.pt")" -gt 800000000
  echo SAM21_READY
) > "$WORK/logs/sam21_download.log" 2>&1 &
SAM21_PID=$!

(
  set -euo pipefail
  # 差別 1：P1 合併權重（sam3.pt 本體 ＋ 訓過的 decoder，merge_ckpt.py 產出）。位元數 fail-closed。
  rclone copyto "$P1_SRC" "$CKPT/merged_p1_A.pt"
  test "$(stat -c%s "$CKPT/merged_p1_A.pt")" -eq "$P1_BYTES" || { echo "merged_p1_A.pt 位元數不符"; exit 1; }
  rclone copyto "${P1_SRC}.provenance.json" "$WORK/merged_p1_A.provenance.json" || true
  echo P1_READY
) > "$WORK/logs/p1_download.log" 2>&1 &
P1_PID=$!

(
  set -euo pipefail
  # 官方 sam3.pt 仍要拉：selftest 不用它，但 sha256 記錄與 provenance 要對得上（G16 gDrive 優先）
  if rclone copyto "$GDRIVE/4_models/pretrained/sam3.pt" "$CKPT/sam3.pt" 2>/dev/null \
     && echo "$SAM3_CKPT_SHA256  $CKPT/sam3.pt" | sha256sum -c - >/dev/null 2>&1; then
    echo "sam3.pt 取自 gDrive 備援（sha256 已驗）" > "$WORK/SAM3_WEIGHT_SOURCE.txt"
  else
    rm -f "$CKPT/sam3.pt"
    curl -fL -H "Authorization: Bearer $HF_TOKEN" \
      -o "$CKPT/sam3.pt" "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt"
    echo "$SAM3_CKPT_SHA256  $CKPT/sam3.pt" | sha256sum -c -
    echo "official gated HF (sha256 $SAM3_CKPT_SHA256)" > "$WORK/SAM3_WEIGHT_SOURCE.txt"
  fi
  echo SAM3_READY
) > "$WORK/logs/sam3_download.log" 2>&1 &
SAM3_PID=$!

# ── 兩個完全隔離的 venv（同 drill；PINNED_ENV=1 從鎖裝）─────────────────────
[ -x "$T1ENV/bin/python" ] || uv venv --python 3.12 "$T1ENV"
[ -d "$SAMURAI/.git" ] || git clone https://github.com/yangchris11/samurai.git "$SAMURAI"
git -C "$SAMURAI" fetch --depth 1 origin "$SAMURAI_SHA"
git -C "$SAMURAI" checkout --detach "$SAMURAI_SHA"
VIRTUAL_ENV="$T1ENV" uv pip install -q torch torchvision --torch-backend=auto
if [ "$PINNED_ENV" = 1 ]; then
  VIRTUAL_ENV="$T1ENV" uv pip install -q -r "$SRC/requirements-t1env.txt"
  VIRTUAL_ENV="$T1ENV" uv pip install -q -e "$SAMURAI/sam2" pytest
else
  VIRTUAL_ENV="$T1ENV" uv pip install -q -e "$SAMURAI/sam2" \
    scipy loguru tqdm pandas pillow opencv-python-headless pytest
fi
"$PY1" -c "import torch,sam2,pandas; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

[ -x "$SAM3ENV/bin/python" ] || uv venv --python 3.12 "$SAM3ENV"
VIRTUAL_ENV="$SAM3ENV" uv pip install -q torch torchvision --torch-backend=auto
if [ "$PINNED_ENV" = 1 ]; then
  VIRTUAL_ENV="$SAM3ENV" uv pip install -q -r "$SRC/requirements-sam3env.txt"
  VIRTUAL_ENV="$SAM3ENV" uv pip install -q \
    "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}"
else
  VIRTUAL_ENV="$SAM3ENV" uv pip install -q \
    "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}" \
    "numpy<2" ftfy==6.1.1 regex einops psutil scipy av pycocotools numba \
    python-rapidjson pandas pillow tqdm opencv-python-headless
fi
VIRTUAL_ENV="$SAM3ENV" uv pip install -q "setuptools<81"   # D086
"$PY3" -c "import torch,sam3,numpy; assert torch.cuda.is_available(); assert numpy.__version__.startswith('1.'); print(torch.__version__, torch.cuda.get_device_name(0))"

for JOB_PID in "$DATA_PID" "$SAM21_PID" "$P1_PID" "$SAM3_PID"; do wait "$JOB_PID"; done
grep -q DATA_READY "$WORK/logs/data_setup.log"
grep -q SAM21_READY "$WORK/logs/sam21_download.log"
grep -q P1_READY "$WORK/logs/p1_download.log"
grep -q SAM3_READY "$WORK/logs/sam3_download.log"

sha256sum "$CKPT/sam2.1_hiera_large.pt" "$CKPT/sam3.pt" "$CKPT/merged_p1_A.pt" > "$WORK/checkpoint_sha256.txt"
git -C "$SAMURAI" rev-parse HEAD > "$WORK/samurai_commit.txt"
uv pip freeze --python "$PY1" > "$WORK/t1_freeze.txt"
uv pip freeze --python "$PY3" > "$WORK/sam3_freeze.txt"
nvidia-smi > "$WORK/nvidia_smi.txt" || true
echo "$(date -Is) ENV_DATA_OK" | tee -a "$WORK/timeline.txt"

sync_progress &
SYNC_PID=$!

# ── 階段 0：finalize 自測（與 ckpt 無關；證明這台機器上的後處理層位元級等於 v078/v090）──
echo "$(date -Is) SELFTEST_START" | tee -a "$WORK/timeline.txt"
mkdir -p "$REPO"/5_outputs/submissions "$REPO"/1_data/raw
rclone copy "$GDRIVE/5_outputs/submissions" "$REPO"/5_outputs/submissions \
  --include 'sub_v023_*.csv' --include 'sub_v012_*.csv' --include 'sub_v049_*.csv' \
  --include 'sub_v056_*.csv' --include 'sub_v078_*.csv' --include 'sub_v090_*.csv' --transfers 8
rclone copy "$GDRIVE/5_outputs/rankb_robust_test75_20260822/run/full_sam3" \
  "$REPO"/5_outputs/rankb_robust_test75_20260822/run/full_sam3 \
  --include 'submission.csv' --transfers 4
rclone copy "$GDRIVE/5_outputs/cropwindow_ensemble_20260831" \
  "$REPO"/5_outputs/cropwindow_ensemble_20260831 --max-depth 1 \
  --include 'A_main_merged.csv' --include 'B_main_merged.csv' --include 'C_main_merged.csv' \
  --include 'A_source_merged.csv' --include 'A_full_sam3.csv' --transfers 8
cp "$SELFTEST_SAMPLE" "$REPO"/1_data/raw/sample_submisson.csv   # G11：selftest 只吃歷史 test75
"$PY1" "$SRC/finalize_submission.py" --selftest 2>&1 | tee "$WORK/logs/selftest.log"
grep -q "自測 (c) 通過" "$WORK/logs/selftest.log" || die "selftest (c) 未通過，停工"
grep -q "自測 (d) 通過" "$WORK/logs/selftest.log" || die "selftest (d) 未通過，停工"
echo "$(date -Is) SELFTEST_OK" | tee -a "$WORK/timeline.txt"

# ── 階段 1：dry-run（BLOCK=0 才准 --execute）──────────────────────────────
echo "$(date -Is) DRYRUN_START" | tee -a "$WORK/timeline.txt"
"$PY1" "$SRC/run_ranking_b.py" \
  --profile "$PROFILE" --allow-offline-two-pass \
  --frames-root "$FRAMES" --sample "$SAMPLE" \
  --work-dir "$WORK/run" --out "$WORK/p1_test75_v090chain_A.csv" \
  --sam3-python "$PY3" --samurai-python "$PY1" \
  --sam3-ckpt "$CKPT/merged_p1_A.pt" \
  --samurai-dir "$SAMURAI" --samurai-ckpt "$CKPT/sam2.1_hiera_large.pt" \
  --sam3-source-revision "$SAM3_SHA" --samurai-source-revision "$SAMURAI_SHA" \
  2>&1 | tee "$WORK/logs/dryrun.log"
grep -q "BLOCK=0" "$WORK/logs/dryrun.log" || die "dry-run 有 BLOCK，停工（見 logs/dryrun.log）"

# ── 階段 2：正式執行（差別 2：--sam3-ckpt ＝ merged_p1_A.pt）────────────────
echo "$(date -Is) INFERENCE_START" | tee -a "$WORK/timeline.txt"
"$PY1" "$SRC/run_ranking_b.py" \
  --profile "$PROFILE" --allow-offline-two-pass \
  --frames-root "$FRAMES" --sample "$SAMPLE" \
  --work-dir "$WORK/run" --out "$WORK/p1_test75_v090chain_A.csv" \
  --sam3-python "$PY3" --samurai-python "$PY1" \
  --sam3-ckpt "$CKPT/merged_p1_A.pt" \
  --samurai-dir "$SAMURAI" --samurai-ckpt "$CKPT/sam2.1_hiera_large.pt" \
  --sam3-source-revision "$SAM3_SHA" --samurai-source-revision "$SAMURAI_SHA" \
  --execute 2>&1 | tee "$WORK/logs/execute.log"
test -s "$WORK/p1_test75_v090chain_A.csv"
echo "$(date -Is) INFERENCE_DONE" | tee -a "$WORK/timeline.txt"

# ── 階段 3：對 sub_v090 的足跡（差別 3：只記錄，不 die）─────────────────────
"$PY1" - "$WORK/p1_test75_v090chain_A.csv" "$REPO/5_outputs/submissions/$REF_SUB" \
  > "$WORK/p1_vs_v090.txt" 2>&1 <<'PY' || true
import csv, sys
def load(p):
    with open(p, newline="") as fh:
        return {r["ID"]: tuple(float(r[c]) for c in ("x","y","width","height")) for r in csv.DictReader(fh)}
a, b = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(a) & set(b))
def iou(p, q):
    ax2, ay2, bx2, by2 = p[0]+p[2], p[1]+p[3], q[0]+q[2], q[1]+q[3]
    iw = max(0.0, min(ax2,bx2)-max(p[0],q[0])); ih = max(0.0, min(ay2,by2)-max(p[1],q[1]))
    inter = iw*ih; union = p[2]*p[3] + q[2]*q[3] - inter
    return inter/union if union > 0 else 0.0
seqs = {}
for k in common:
    seqs.setdefault(k.rsplit("_",1)[0], []).append(iou(a[k], b[k]))
same = sum(1 for k in common if a[k] == b[k])
print(f"共同列 {len(common)}／{len(a)}；逐列相同 {same} ({100*same/len(common):.1f}%)")
print(f"對 v090 逐序列平均 IoU {sum(sum(v)/len(v) for v in seqs.values())/len(seqs):.5f}")
print("改動最大的 15 支（IoU 越低＝改越多）:")
for m, s in sorted((sum(v)/len(v), s) for s, v in seqs.items())[:15]:
    print(f"  {s}: {m:.4f}")
PY
cat "$WORK/p1_vs_v090.txt"
echo "$(date -Is) STAGEC_DONE" | tee -a "$WORK/timeline.txt"
