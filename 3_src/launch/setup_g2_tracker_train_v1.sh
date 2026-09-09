#!/usr/bin/env bash
# G2 = tracker-only 真訓練＋災難檢查（G1 PASS 後續關；A10 單卡）。
# 五階段：prep(405 frames) → train_g2(3000 步) → merge_ckpt → track_t1 smoke
# （trained vs frozen × 6 支）→ smoke_compare（災難閘）→ g2_summary.json。
#
# 啟動前置假設同 G1 腳本（缺任一 fail-fast）：cloud-init selfkill 已掛、
# repo 在 /home/ubuntu/hsot_repo、HF_TOKEN 已 export。
#
# ── SELFKILL：rearm 330 分（不是 250）────────────────────────────────────
# 逐項時間算術（G1 實測外推；寫明讓下一個人可重推）：
#   prep+下載並行     ~15 分
#   train 3000 步     ~185-195 分（G1 的 18 steps/min 是 10 clips 全預載 RAM 的
#                     數字；G2 訓練集 lazy 逐 step 讀 8 張 JPEG ≈ +0.3-0.5s/步）
#   holdout eval      ~35-40 分（15 次 × 80 clips × ~1.5-2s forward——
#                     ⚠️ 這一項在 250 分的原估裡完全沒入帳）
#   merge+verify      ~10 分（1.6GB 重載逐鍵 torch.equal）
#   smoke 2×6 支      ~20-25 分（~5000 幀 × 0.23s/幀，演練 #3 外推）
#   合計              ~270-285 分 ⇒ rearm 330 留 ~50 分餘裕。
# 250 會在完工前幾分鐘砍機器＝08-07 事故的形狀（差 3 分鐘被自毀砍掉）。
# 單一計時器慣例同 G1（q100）：cloud-init backstop 是唯一保險、不自掛第二顆。
set -euo pipefail

export PYTHONUNBUFFERED=1
export PATH="/home/ubuntu/.local/bin:$PATH"
export RCLONE_CONFIG=/home/ubuntu/rclone.conf

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DEST="$GDRIVE/5_outputs/g2_tracker_train_20260829"

die() { echo "FATAL: $*" >&2; exit 1; }

: "${HF_TOKEN:?HF_TOKEN is required (gated facebook/sam3); export it before running}"
sudo test -x /root/rearm_selfkill.sh \
  || die "缺 /root/rearm_selfkill.sh：cloud-init selfkill backstop 未就位"
sudo /root/rearm_selfkill.sh 330   # 見上方逐項算術；開機端武裝的初始 90 分不夠

SRC=/home/ubuntu/hsot_repo          # D093：不可叫 hsot（與套件同名撞 namespace）
WORK=/home/ubuntu/g2_tracker_train
RAW=/home/ubuntu/train405_raw_archive
FRAMES=/home/ubuntu/train405_fc
CKPT=/home/ubuntu/ckpt
SAM3ENV=/home/ubuntu/sam3env
PY3="$SAM3ENV/bin/python"
SAM3_SHA=96914d2425f90a64f45ca977c2b5165418099543
SAM3_CKPT_SHA256=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
SYNC_PID=""

test -d "$SRC/3_src" || die "repo 不在 $SRC（開機流程應先放好）"

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

# ── 資料（背景並行；prep 對齊邏輯照搬 G1 腳本——新機器要重 prep）─────────
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

# ── 環境（隔離 venv 同 G1）──────────────────────────────────────────────
[ -x "$SAM3ENV/bin/python" ] || uv venv --python 3.12 "$SAM3ENV"
VIRTUAL_ENV="$SAM3ENV" uv pip install -q torch torchvision --torch-backend=auto
VIRTUAL_ENV="$SAM3ENV" uv pip install -q \
  "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}" \
  "numpy<2" ftfy==6.1.1 regex einops psutil scipy av pycocotools numba \
  python-rapidjson pandas pillow tqdm opencv-python-headless pytest
VIRTUAL_ENV="$SAM3ENV" uv pip install -q "setuptools<81"   # D086
"$PY3" -c "import torch,sam3,numpy,pandas; assert torch.cuda.is_available(); assert numpy.__version__.startswith('1.'); print(torch.__version__, torch.cuda.get_device_name(0))"

for JOB_PID in $DATA_PIDS "$GT_PID" "$SAM3_PID"; do
  wait "$JOB_PID"
done
grep -q GT_READY "$WORK/logs/gt_download.log"
grep -q SAM3_READY "$WORK/logs/sam3_download.log"

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

echo "$(date -Is) UNITTEST_START" | tee -a "$WORK/timeline.txt"
(cd "$SRC" && "$PY3" -m pytest -q 3_src/test_train_tracker_g1.py 3_src/test_train_tracker_g2.py) \
  2>&1 | tee "$WORK/logs/pytest_g2.log"

# ── 階段 1：訓練（產物邊跑邊回傳）───────────────────────────────────────
sync_progress &
SYNC_PID=$!
echo "$(date -Is) TRAIN_START" | tee -a "$WORK/timeline.txt"
# rc：0=GO、2=NO_GO、3=ABORT 是有效判決；4=ERROR_INPLACE 與其他非零是真失敗。
TRAIN_RC=0
"$PY3" "$SRC/3_src/train_tracker_g1/train_g2.py" \
  --clips-json /home/ubuntu/hard_clips_vis_rednir.json \
  --gt-csv /home/ubuntu/2026training.csv \
  --frames-root "$FRAMES" \
  --sam3-ckpt "$CKPT/sam3.pt" \
  --out-dir "$WORK/run" \
  --train-folds 0,1,2,3 --holdout-folds 4 \
  --steps 3000 --eval-every 200 --t 8 --stride 2 --image-size 1008 \
  --lr 1e-4 --seed 42 --device cuda \
  2>&1 | tee "$WORK/logs/train_g2.log" || TRAIN_RC=$?
if [ "$TRAIN_RC" -ne 0 ] && [ "$TRAIN_RC" -ne 2 ] && [ "$TRAIN_RC" -ne 3 ]; then
  die "train_g2.py 異常結束 rc=$TRAIN_RC（非 GO/NO_GO/ABORT 判決）"
fi
test -s "$WORK/run/g2_verdict.json"
test -s "$WORK/run/tracker_g2_best.pt"
test -s "$WORK/run/train_log.jsonl"
echo "$(date -Is) TRAIN_DONE rc=$TRAIN_RC" | tee -a "$WORK/timeline.txt"

# ── 階段 2+3：merge ＋ 同構性 smoke ＋ 災難檢查 ─────────────────────────
# ABORT（rc=3）時跳過：best ckpt ≈ step0 噪聲，smoke 25 分鐘量不到東西。
SMOKE_RC=-1
if [ "$TRAIN_RC" -ne 3 ]; then
  echo "$(date -Is) MERGE_START" | tee -a "$WORK/timeline.txt"
  "$PY3" "$SRC/3_src/train_tracker_g1/merge_ckpt.py" \
    --base "$CKPT/sam3.pt" \
    --tracker-ckpt "$WORK/run/tracker_g2_best.pt" \
    --out "$WORK/run/merged_g2.pt" \
    2>&1 | tee "$WORK/logs/merge.log"

  "$PY3" "$SRC/3_src/train_tracker_g1/train_g2.py" \
    --clips-json /home/ubuntu/hard_clips_vis_rednir.json \
    --gt-csv /home/ubuntu/2026training.csv \
    --frames-root "$FRAMES" \
    --out-dir "$WORK/run" \
    --smoke-plan-out "$WORK/run/smoke_plan.json"
  test -s "$WORK/run/smoke_plan.json"
  test -s "$WORK/run/smoke_plan.json.seqlist.txt"

  # track_t1 sam3 腿旗標照 rankB_robust profile（依據＝configs/ranking_profiles.json
  # 的 tracking 段 ＋ run_ranking_b.py:157-169 的旗標翻譯：--sam3-eval 有、
  # --sam3-samurai 無）。不帶 --sample-csv（6 支 subset 不做 exact-set 驗證）。
  echo "$(date -Is) SMOKE_START" | tee -a "$WORK/timeline.txt"
  for LEG in trained frozen; do
    if [ "$LEG" = trained ]; then LEG_CKPT="$WORK/run/merged_g2.pt"; else LEG_CKPT="$CKPT/sam3.pt"; fi
    "$PY3" "$SRC/3_src/track_t1.py" \
      --backend sam3 --sam3-version sam3 --sam3-eval \
      --sam3-ckpt "$LEG_CKPT" \
      --frames-root "$FRAMES" \
      --seq-list "$WORK/run/smoke_plan.json.seqlist.txt" \
      --gt-csv /home/ubuntu/2026training.csv \
      --out-dir "$WORK/smoke_$LEG" \
      --source-revision "$SAM3_SHA" \
      --device cuda:0 \
      2>&1 | tee "$WORK/logs/smoke_${LEG}.log"
    test -s "$WORK/smoke_$LEG/submission.csv"
  done

  SMOKE_RC=0
  "$PY3" "$SRC/3_src/train_tracker_g1/smoke_compare.py" \
    --trained-csv "$WORK/smoke_trained/submission.csv" \
    --frozen-csv "$WORK/smoke_frozen/submission.csv" \
    --gt-csv /home/ubuntu/2026training.csv \
    --plan-json "$WORK/run/smoke_plan.json" \
    --out "$WORK/run/smoke_compare.json" \
    2>&1 | tee "$WORK/logs/smoke_compare.log" || SMOKE_RC=$?
  # rc 5=CATASTROPHE 也是有效判決（json 為準）；其他非零才是真失敗。
  if [ "$SMOKE_RC" -ne 0 ] && [ "$SMOKE_RC" -ne 5 ]; then
    die "smoke_compare.py 異常結束 rc=$SMOKE_RC"
  fi
  test -s "$WORK/run/smoke_compare.json"
  echo "$(date -Is) SMOKE_DONE rc=$SMOKE_RC" | tee -a "$WORK/timeline.txt"
else
  echo "$(date -Is) SMOKE_SKIPPED（train ABORT）" | tee -a "$WORK/timeline.txt"
fi

# ── 匯總（兩個分支都要寫）───────────────────────────────────────────────
"$PY3" - "$WORK" "$TRAIN_RC" "$SMOKE_RC" <<'PYEOF'
import json, sys
from pathlib import Path
work, train_rc, smoke_rc = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
train_v = json.loads((work / "run/g2_verdict.json").read_text())
smoke_p = work / "run/smoke_compare.json"
smoke_v = json.loads(smoke_p.read_text()) if smoke_p.exists() else None
summary = {
    "train_verdict": train_v.get("verdict"),
    "train_rc": train_rc,
    "smoke_verdict": (smoke_v or {}).get("verdict", "SKIPPED"),
    "smoke_rc": smoke_rc if smoke_rc >= 0 else None,  # −1＝跳過，不是 exit code
    "best_iou": train_v.get("best_iou"), "best_step": train_v.get("best_step"),
    "iou_gain": train_v.get("iou_gain"),
    "catastrophe_sequences": (smoke_v or {}).get("catastrophe_sequences", []),
}
(work / "run/g2_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, ensure_ascii=False))
PYEOF
test -s "$WORK/run/g2_summary.json"
echo "$(date -Is) G2_DONE" | tee -a "$WORK/timeline.txt"
cat "$WORK/run/g2_summary.json"
