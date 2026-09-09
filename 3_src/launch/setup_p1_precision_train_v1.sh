#!/usr/bin/env bash
# P1 精度線（D105，2026-09-02）：uniform 取樣的 tracker 微調 → Stage A/B 判準 → GO 才跑 test75 v090 全鏈。
#
# 六階段（每階段 fail-closed；產物邊跑邊 rclone 回 gDrive）：
#   0 環境＋資料（sam3env／t1env＋SAMURAI／405 假色 zip／test75 tar／權重／405 zero-shot 預測）
#   1 prep_oof405_frames（405 支、167,174 行 contract）→ build_uniform_clips → pytest → 凍結範圍自檢
#   2 train_p1 ×ARMS（A=decoder／B=tracker；A10 只跑 A）→ Stage A verdict（PASS_A／FAIL_A／ABORT）
#   3 Stage B smoke：frozen 腿一次 ＋ 每個 PASS_A arm 一次（fold-4 代表性序列）→ precision_compare
#   4 arm 選擇（precision_compare --select）
#   5 Stage C（僅 GO）：run_ranking_b --profile rankB_deliver_v090 --sam3-ckpt <merged> → LB 候選 CSV
#
# 啟動前置假設同 G2 腳本（缺任一 fail-fast）：cloud-init selfkill 已掛、repo 在 /home/ubuntu/hsot_repo、
# HF_TOKEN 已 export、rclone.conf 在 /home/ubuntu/rclone.conf。
#
# ── SELFKILL 算術（G2 慣例：逐項寫明，讓下一個人可重推）──────────────────────
#   A100（兩 arm）：env+data+prep 25 ＋ 每 arm（train 3000 步 ~75 ＋ holdout eval 12×120 clips ~18）×2
#                 ＋ smoke 3 腿 × ~6 ＋ Stage C v090 全鏈 ~130 ＋ 收尾 10 ＝ ~375 ⇒ rearm 510（×1.35 緩衝）
#   A10（單 arm A）：25 ＋（190 ＋ 45）＋ 2×15 ＋ Stage C ~270 ＋ 10 ＝ ~570 ⇒ rearm 720
#   （A10 數字取 G2 腳本實測外推；A100 以 SAM3 推論 2.8× 的專案實測折算，訓練保守取 2.5×）
#   單一計時器慣例同 G1/G2：cloud-init backstop 是唯一保險、不自掛第二顆。
set -euo pipefail

export PYTHONUNBUFFERED=1
export PATH="/home/ubuntu/.local/bin:$PATH"
export RCLONE_CONFIG=/home/ubuntu/rclone.conf

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
# 09-05：P1_DEST 可覆寫產物資料夾（6000 步延長訓練不可覆寫 09-04 的 3000 步產物）
DEST="$GDRIVE/5_outputs/${P1_DEST:-p1_precision_train_20260902}"

die() { echo "FATAL: $*" >&2; exit 1; }

: "${HF_TOKEN:?HF_TOKEN is required (gated facebook/sam3); export it before running}"
sudo test -x /root/rearm_selfkill.sh \
  || die "缺 /root/rearm_selfkill.sh：cloud-init selfkill backstop 未就位"

# ── GPU 閘門（08-31 壞卡事故：ECC 未修正錯誤會讓 track_t1 半途 SIGABRT，極易誤判成程式問題）──
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
ECC="$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader | head -1 | tr -d ' ')"
case "$ECC" in
  0|"[N/A]"|"N/A"|"") ;;
  *) die "GPU ECC uncorrected=$ECC（壞卡，terminate 換機）";;
esac
if [[ "$GPU_NAME" == *A100* || "$GPU_NAME" == *H100* ]]; then
  ARMS="${ARMS:-decoder,tracker}"; REARM="${REARM_MIN:-510}"
else
  ARMS="${ARMS:-decoder}"; REARM="${REARM_MIN:-720}"   # D105：僅 A10 ⇒ 砍為單 arm A
fi
echo "GPU=$GPU_NAME ECC=$ECC ARMS=$ARMS REARM=$REARM"
sudo /root/rearm_selfkill.sh "$REARM"

REPO=/home/ubuntu/hsot_repo         # D093：不可叫 hsot（與套件同名撞 namespace）
SRC="$REPO/3_src"
WORK=/home/ubuntu/p1_precision
RAW=/home/ubuntu/train405_raw_archive
FRAMES=/home/ubuntu/train405_fc
TEST_FRAMES=/home/ubuntu/test_fc
CKPT=/home/ubuntu/ckpt
SAMURAI=/home/ubuntu/samurai
T1ENV=/home/ubuntu/t1env
SAM3ENV=/home/ubuntu/sam3env
PY1="$T1ENV/bin/python"
PY3="$SAM3ENV/bin/python"
SAM3_SHA=96914d2425f90a64f45ca977c2b5165418099543
SAMURAI_SHA=76ba195984892b0d1e3db5d9c9f90bb62175680a
SAM3_CKPT_SHA256=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
SYNC_PID=""

test -d "$SRC" || die "repo 不在 $REPO（開機流程應先放好）"
export PYTHONPATH="$SRC"            # D093／08-31：獨立腳本必須自己帶 PYTHONPATH

sync_progress() {
  while true; do
    rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 \
      --exclude '*.part' --exclude '*.part/**' \
      --exclude 'stageC_run/offline_two_pass/frames_*/**' || true
    sleep 60
  done
}

finish() {
  RUN_RC=$?
  trap - EXIT
  set +e
  if [ -n "$SYNC_PID" ]; then kill "$SYNC_PID" 2>/dev/null; wait "$SYNC_PID" 2>/dev/null; fi
  echo "$(date -Is) finish rc=$RUN_RC" | tee -a "$WORK/finish.txt"
  SYNC_RC=0
  if command -v rclone >/dev/null 2>&1 && [ -f "$RCLONE_CONFIG" ]; then
    rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 \
      --exclude '*.part' --exclude '*.part/**' \
      --exclude 'stageC_run/offline_two_pass/frames_*/**' || SYNC_RC=$?
    if [ "$SYNC_RC" -eq 0 ]; then
      rclone check "$WORK" "$DEST" --one-way \
        --exclude '*.part' --exclude '*.part/**' \
        --exclude 'stageC_run/offline_two_pass/frames_*/**' || SYNC_RC=$?
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

mkdir -p "$WORK/logs" "$WORK/run" "$RAW/training" "$RAW/update" "$FRAMES" "$TEST_FRAMES" "$CKPT"
chmod 600 "$RCLONE_CONFIG"
echo "$(date -Is) SETUP_START gpu=$GPU_NAME arms=$ARMS" | tee "$WORK/timeline.txt"

if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | sudo bash >/dev/null
fi
rclone lsf "$GDRIVE/1_data/raw_archive/training" >/dev/null
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi

# ── 階段 0a：資料（背景並行）──────────────────────────────────────────────
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
  # 405 zero-shot full-frame SAM3 預測：只供 uniform clip 索引的統計欄（mean_iou 等），不進訓練
  rclone copyto "$GDRIVE/5_outputs/oof405_exact_pair_20260827/run/full_sam3/submission.csv" \
    /home/ubuntu/oof405_full_sam3.csv
  test -s /home/ubuntu/oof405_full_sam3.csv
  # Stage C 材料（test75 假色、sample、sub_v090 對照）
  rclone copyto "$GDRIVE/1_data/packed/t1test_fc_75.tar" /home/ubuntu/test_fc_75.tar
  tar -xf /home/ubuntu/test_fc_75.tar -C "$TEST_FRAMES"
  test "$(find "$TEST_FRAMES" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 75
  rclone copyto "$GDRIVE/1_data/raw/sample_submisson.csv" /home/ubuntu/sample_submisson.csv
  mkdir -p "$REPO/5_outputs/submissions"
  rclone copy "$GDRIVE/5_outputs/submissions" "$REPO/5_outputs/submissions" \
    --include 'sub_v090_*.csv' --transfers 4
  test -n "$(ls "$REPO"/5_outputs/submissions/sub_v090_*.csv 2>/dev/null | head -1)"
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
  rclone copyto "$GDRIVE/4_models/pretrained/sam2.1_hiera_large.pt" "$CKPT/sam2.1_hiera_large.pt"
  test "$(stat -c%s "$CKPT/sam2.1_hiera_large.pt")" -gt 800000000
  echo SAM3_READY
) > "$WORK/logs/sam3_download.log" 2>&1 &
SAM3_PID=$!

# ── 階段 0b：環境（sam3env 同 G2；t1env＋SAMURAI 同 G3，Stage C 才用）──────
[ -x "$SAM3ENV/bin/python" ] || uv venv --python 3.12 "$SAM3ENV"
VIRTUAL_ENV="$SAM3ENV" uv pip install -q torch torchvision --torch-backend=auto
VIRTUAL_ENV="$SAM3ENV" uv pip install -q \
  "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}" \
  "numpy<2" ftfy==6.1.1 regex einops psutil scipy av pycocotools numba \
  python-rapidjson pandas pillow tqdm opencv-python-headless pytest
VIRTUAL_ENV="$SAM3ENV" uv pip install -q "setuptools<81"   # D086
"$PY3" -c "import torch,sam3,numpy,pandas; assert torch.cuda.is_available(); assert numpy.__version__.startswith('1.'); print(torch.__version__, torch.cuda.get_device_name(0))"

[ -x "$T1ENV/bin/python" ] || uv venv --python 3.12 "$T1ENV"
[ -d "$SAMURAI/.git" ] || git clone https://github.com/yangchris11/samurai.git "$SAMURAI"
git -C "$SAMURAI" fetch --depth 1 origin "$SAMURAI_SHA"
git -C "$SAMURAI" checkout --detach "$SAMURAI_SHA"
VIRTUAL_ENV="$T1ENV" uv pip install -q torch torchvision --torch-backend=auto
VIRTUAL_ENV="$T1ENV" uv pip install -q -e "$SAMURAI/sam2" \
  scipy loguru tqdm pandas pillow opencv-python-headless
"$PY1" -c "import torch, sam2; assert torch.cuda.is_available()"

for JOB_PID in $DATA_PIDS "$GT_PID" "$SAM3_PID"; do
  wait "$JOB_PID"
done
grep -q GT_READY "$WORK/logs/gt_download.log"
grep -q SAM3_READY "$WORK/logs/sam3_download.log"
echo "$(date -Is) ENV_DATA_OK" | tee -a "$WORK/timeline.txt"

# ── 階段 1：prep → uniform clips → pytest → 凍結範圍自檢 ─────────────────
"$PY3" "$SRC/prep/prep_oof405_frames.py" \
  --zip-root "$RAW" --gt /home/ubuntu/2026training.csv \
  --frames-out "$FRAMES" --contracts-out "$WORK/contracts" --hash-zips
test "$(find "$FRAMES" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 405
test "$(($(wc -l < "$WORK/contracts/sample_contract.csv") - 1))" -eq 167174

"$PY3" "$SRC/build_uniform_clips.py" \
  --frame-contract "$WORK/contracts/frame_contract.csv" \
  --gt-csv /home/ubuntu/2026training.csv \
  --pred-csv /home/ubuntu/oof405_full_sam3.csv \
  --out "$WORK/run/uniform_clips_all.json" 2>&1 | tee "$WORK/logs/build_uniform_clips.log"
test -s "$WORK/run/uniform_clips_all.json"

sha256sum "$CKPT/sam3.pt" > "$WORK/checkpoint_sha256.txt"
uv pip freeze --python "$PY3" > "$WORK/sam3_freeze.txt"
uv pip freeze --python "$PY1" > "$WORK/t1_freeze.txt"
nvidia-smi > "$WORK/nvidia_smi.txt" || true

echo "$(date -Is) UNITTEST_START" | tee -a "$WORK/timeline.txt"
(cd "$REPO" && "$PY3" -m pytest -q 3_src/test_train_tracker_g1.py 3_src/test_train_tracker_g2.py \
   3_src/test_train_tracker_p1.py) 2>&1 | tee "$WORK/logs/pytest_p1.log"

# 凍結範圍自檢（advisor 09-02：本機無 torch，這道 assert 只能在機上第一次 backward 前驗）
IFS=',' read -r -a ARM_LIST <<< "$ARMS"
for SCOPE in "${ARM_LIST[@]}"; do
  "$PY3" - "$CKPT/sam3.pt" "$SCOPE" <<'PYEOF'
import sys
from train_tracker_g1.sot_trainable import build_trainable_tracker_scoped, SCOPE_PARAM_RANGES
ckpt, scope = sys.argv[1], sys.argv[2]
_, tracker, n = build_trainable_tracker_scoped(ckpt, device="cuda", scope=scope)
lo, hi = SCOPE_PARAM_RANGES[scope]
assert lo < n < hi, (scope, n)
frozen_train = [name for name, m in tracker.named_children()
                if name != "backbone" and m.training and scope == "decoder" and name != "sam_mask_decoder"]
assert not frozen_train, f"decoder scope 下仍在 train mode 的凍結子模組：{frozen_train}"
print(f"[selfcheck] scope={scope} trainable={n/1e6:.2f}M OK")
PYEOF
done
echo "$(date -Is) SELFCHECK_OK" | tee -a "$WORK/timeline.txt"

# ── 階段 2：訓練（每個 arm 一次；產物邊跑邊回傳）──────────────────────────
sync_progress &
SYNC_PID=$!
declare -A ARM_LABEL=( [decoder]=A [tracker]=B )
declare -A TRAIN_RC_OF
for SCOPE in "${ARM_LIST[@]}"; do
  ARM="${ARM_LABEL[$SCOPE]}"
  echo "$(date -Is) TRAIN_START arm=$ARM scope=$SCOPE" | tee -a "$WORK/timeline.txt"
  RC=0
  "$PY3" "$SRC/train_tracker_g1/train_p1.py" \
    --clips-json "$WORK/run/uniform_clips_all.json" \
    --gt-csv /home/ubuntu/2026training.csv \
    --frames-root "$FRAMES" \
    --sam3-ckpt "$CKPT/sam3.pt" \
    --out-dir "$WORK/run" \
    --arm "$ARM" --trainable-scope "$SCOPE" \
    --train-folds 0,1,2,3 --holdout-folds 4 \
    --steps "${STEPS:-3000}" --eval-every 250 --holdout-max 120 \
    --t 8 --stride 2 --image-size 1008 --lr 1e-4 --seed 42 --device cuda \
    2>&1 | tee "$WORK/logs/train_p1_${ARM}.log" || RC=$?
  # rc：0=PASS_A、2=FAIL_A、3=ABORT 是有效判決；4=ERROR_INPLACE 與其他非零是真失敗。
  if [ "$RC" -ne 0 ] && [ "$RC" -ne 2 ] && [ "$RC" -ne 3 ]; then
    die "train_p1.py arm=$ARM 異常結束 rc=$RC"
  fi
  test -s "$WORK/run/p1_${ARM}_${SCOPE}_stageA.json"
  TRAIN_RC_OF[$ARM]=$RC
  echo "$(date -Is) TRAIN_DONE arm=$ARM rc=$RC" | tee -a "$WORK/timeline.txt"
done

# ── 階段 3：Stage B smoke（frozen 一次 ＋ 每個 PASS_A arm 一次）──────────
"$PY3" "$SRC/train_tracker_g1/train_p1.py" \
  --clips-json "$WORK/run/uniform_clips_all.json" \
  --gt-csv /home/ubuntu/2026training.csv \
  --frames-root "$FRAMES" --out-dir "$WORK/run" --holdout-folds 4 \
  --smoke-plan-out "$WORK/run/stageB_plan.json"
test -s "$WORK/run/stageB_plan.json.seqlist.txt"

PASS_ARMS=()
for SCOPE in "${ARM_LIST[@]}"; do
  ARM="${ARM_LABEL[$SCOPE]}"
  [ "${TRAIN_RC_OF[$ARM]}" -eq 0 ] && PASS_ARMS+=("$ARM:$SCOPE")
done
echo "$(date -Is) STAGEB_PLAN_OK pass_arms=${PASS_ARMS[*]:-none}" | tee -a "$WORK/timeline.txt"

# track_t1 sam3 腿旗標照 rankB profile（--sam3-eval 有、--sam3-samurai 無），同 G2 smoke。
run_leg() {  # $1=leg 名  $2=ckpt
  "$PY3" "$SRC/track_t1.py" \
    --backend sam3 --sam3-version sam3 --sam3-eval \
    --sam3-ckpt "$2" \
    --frames-root "$FRAMES" \
    --seq-list "$WORK/run/stageB_plan.json.seqlist.txt" \
    --gt-csv /home/ubuntu/2026training.csv \
    --out-dir "$WORK/smoke_$1" \
    --source-revision "$SAM3_SHA" \
    --device cuda:0 2>&1 | tee "$WORK/logs/smoke_$1.log"
  test -s "$WORK/smoke_$1/submission.csv"
}

COMPARE_JSONS=()
if [ "${#PASS_ARMS[@]}" -gt 0 ]; then
  echo "$(date -Is) SMOKE_START" | tee -a "$WORK/timeline.txt"
  run_leg frozen "$CKPT/sam3.pt"
  for PAIR in "${PASS_ARMS[@]}"; do
    ARM="${PAIR%%:*}"; SCOPE="${PAIR##*:}"
    "$PY3" "$SRC/train_tracker_g1/merge_ckpt.py" \
      --base "$CKPT/sam3.pt" \
      --tracker-ckpt "$WORK/run/tracker_p1_${ARM}_${SCOPE}_best.pt" \
      --out "$WORK/run/merged_p1_${ARM}.pt" 2>&1 | tee "$WORK/logs/merge_${ARM}.log"
    run_leg "trained_${ARM}" "$WORK/run/merged_p1_${ARM}.pt"
    GAIN="$("$PY3" -c "import json,sys; print(json.load(open(sys.argv[1]))['clip_gain'])" \
             "$WORK/run/p1_${ARM}_${SCOPE}_stageA.json")"
    CRC=0
    "$PY3" "$SRC/train_tracker_g1/precision_compare.py" \
      --trained-csv "$WORK/smoke_trained_${ARM}/submission.csv" \
      --frozen-csv "$WORK/smoke_frozen/submission.csv" \
      --gt-csv /home/ubuntu/2026training.csv \
      --plan-json "$WORK/run/stageB_plan.json" \
      --clip-gain "$GAIN" --arm "$ARM" \
      --out "$WORK/run/p1_${ARM}_stageB.json" 2>&1 | tee "$WORK/logs/compare_${ARM}.log" || CRC=$?
    # rc 0/2/5/6/7 皆為有效判決；其他才是真失敗
    case "$CRC" in 0|2|5|6|7) ;; *) die "precision_compare arm=$ARM 異常 rc=$CRC";; esac
    test -s "$WORK/run/p1_${ARM}_stageB.json"
    COMPARE_JSONS+=("$WORK/run/p1_${ARM}_stageB.json")
  done
  echo "$(date -Is) SMOKE_DONE" | tee -a "$WORK/timeline.txt"
else
  echo "$(date -Is) SMOKE_SKIPPED（無 PASS_A arm）" | tee -a "$WORK/timeline.txt"
fi

# ── 階段 4：arm 選擇 ──────────────────────────────────────────────────────
CHOSEN=""
if [ "${#COMPARE_JSONS[@]}" -gt 0 ]; then
  SRC_RC=0
  "$PY3" "$SRC/train_tracker_g1/precision_compare.py" \
    --select "${COMPARE_JSONS[@]}" --out "$WORK/run/p1_selection.json" || SRC_RC=$?
  case "$SRC_RC" in 0|2) ;; *) die "arm 選擇異常 rc=$SRC_RC";; esac
  CHOSEN="$("$PY3" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['chosen'] or '')" \
             "$WORK/run/p1_selection.json")"
fi
echo "$(date -Is) SELECTION chosen=${CHOSEN:-none}" | tee -a "$WORK/timeline.txt"

# ── 階段 5：Stage C（僅 GO）：v090 全鏈、只換 --sam3-ckpt ──────────────────
if [ -n "$CHOSEN" ]; then
  echo "$(date -Is) STAGEC_START arm=$CHOSEN" | tee -a "$WORK/timeline.txt"
  "$PY1" "$SRC/run_ranking_b.py" \
    --profile rankB_deliver_v090 --allow-offline-two-pass \
    --frames-root "$TEST_FRAMES" --sample /home/ubuntu/sample_submisson.csv \
    --work-dir "$WORK/stageC_run" --out "$WORK/p1_test75_v090chain_${CHOSEN}.csv" \
    --sam3-python "$PY3" --samurai-python "$PY1" \
    --sam3-ckpt "$WORK/run/merged_p1_${CHOSEN}.pt" \
    --samurai-dir "$SAMURAI" --samurai-ckpt "$CKPT/sam2.1_hiera_large.pt" \
    --execute 2>&1 | tee "$WORK/logs/stageC_execute.log"
  test -s "$WORK/p1_test75_v090chain_${CHOSEN}.csv"
  V090="$(ls "$REPO"/5_outputs/submissions/sub_v090_*.csv | head -1)"
  "$PY1" - "$WORK/p1_test75_v090chain_${CHOSEN}.csv" "$V090" \
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
print("改動最大的 10 支（IoU 越低＝改越多）:")
for m, s in sorted((sum(v)/len(v), s) for s, v in seqs.items())[:10]:
    print(f"  {s}: {m:.4f}")
PY
  cat "$WORK/p1_vs_v090.txt"
  echo "$(date -Is) STAGEC_DONE" | tee -a "$WORK/timeline.txt"
else
  echo "$(date -Is) STAGEC_SKIPPED（no GO arm）" | tee -a "$WORK/timeline.txt"
fi

# ── 匯總（所有分支都要寫）─────────────────────────────────────────────────
"$PY3" - "$WORK" "$ARMS" "${CHOSEN:-}" <<'PYEOF'
import json, sys, glob
from pathlib import Path
work, arms, chosen = Path(sys.argv[1]), sys.argv[2], sys.argv[3] or None
summary = {"arms": arms.split(","), "chosen": chosen, "stageA": {}, "stageB": {}}
for p in sorted(glob.glob(str(work / "run/p1_*_stageA.json"))):
    d = json.loads(Path(p).read_text()); summary["stageA"][d.get("arm", Path(p).stem)] = {
        "verdict": d["verdict"], "clip_gain": d.get("clip_gain"), "best_step": d.get("best_step"),
        "iou_step0": d.get("iou_step0"), "best_iou": d.get("best_iou")}
for p in sorted(glob.glob(str(work / "run/p1_*_stageB.json"))):
    d = json.loads(Path(p).read_text()); summary["stageB"][d.get("arm", Path(p).stem)] = {
        "verdict": d["verdict"], "pooled_delta": d["pooled_delta"], "worst_delta": d["worst_delta"],
        "auc50_delta": d["diagnostics"]["auc50_delta"],
        "tracked_iou": [d["diagnostics"]["tracked_mean_iou_frozen"], d["diagnostics"]["tracked_mean_iou_trained"]]}
sel = work / "run/p1_selection.json"
summary["selection"] = json.loads(sel.read_text()) if sel.exists() else None
csvs = glob.glob(str(work / "p1_test75_v090chain_*.csv"))
summary["stageC_csv"] = csvs[0] if csvs else None
(work / "run/p1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, ensure_ascii=False))
PYEOF
test -s "$WORK/run/p1_summary.json"
echo "$(date -Is) P1_DONE" | tee -a "$WORK/timeline.txt"
