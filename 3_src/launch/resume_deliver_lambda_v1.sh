#!/usr/bin/env bash
# 9/7 中途失敗後的續跑（09-05 稽核 G17）。
#
# 為什麼需要這支：drill 中途壞卡／spot 回收／ECC SIGABRT 後換新機重跑，**不會 BLOCK，
# 會靜默從零再跑 4.7 小時**——因為沒有任何步驟把 gDrive 上已完成的 run/ 拉回來，
# 而 run_ranking_b.py 看到空的 work-dir 就當全新開始。三天窗口吃得下一次，吃不下兩次。
#
# 本腳本＝drill 的階段 2 加上「先把 run/ 拉回來」，並顯式帶 --resume-existing-work。
# **旗標與 drill 逐字相同**（差一個字就會產生不同的計畫，等於不是續跑而是另一次執行）。
#
# 前置：環境／資料／權重都已就緒（亦即 drill 的階段 0 跑完過）。若是全新機器，
# 先跑 drill 到 SETUP 完成、或手動重建兩個 venv 與 $CKPT/$FRAMES，再跑本腳本。
#
# 用法：
#   DRILL_DEST=deliver_v090_drill_20260907 bash resume_deliver_lambda_v1.sh
#   DELIVER_PROFILE=rankB_deliver_v078 DRILL_DEST=... bash resume_deliver_lambda_v1.sh
#
# ⚠️ D078 已消滅「禁止 resume」的舊限制（那條的依據是 fallback 可 resume 會污染結果，
#    而 fallback 現在預設 exit 2、不可 resume）。PROVENANCE:445-447 的舊句應同步更新。
set -euo pipefail
trap 'echo "死於第 $LINENO 行" >&2' ERR

export PYTHONUNBUFFERED=1
export PATH="/home/ubuntu/.local/bin:$PATH"
export RCLONE_CONFIG=/home/ubuntu/rclone.conf

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DRILL_DEST="${DRILL_DEST:?請指定 DRILL_DEST（要續跑的那次演練／正式跑的資料夾名）}"
DEST="$GDRIVE/5_outputs/$DRILL_DEST"
REPO=/home/ubuntu/hsot_repo          # D093：不可叫 hsot（與套件同名撞 namespace）
SRC="$REPO/3_src"
WORK="/home/ubuntu/$DRILL_DEST"
FRAMES=/home/ubuntu/test_fc
CKPT=/home/ubuntu/ckpt
SAMURAI=/home/ubuntu/samurai
PY1=/home/ubuntu/t1env/bin/python
PY3=/home/ubuntu/sam3env/bin/python
SAMPLE=/home/ubuntu/sample_submisson.csv
SAMURAI_SHA=76ba195984892b0d1e3db5d9c9f90bb62175680a
SAM3_SHA=96914d2425f90a64f45ca977c2b5165418099543
PROFILE="${DELIVER_PROFILE:-rankB_deliver_v090}"
SYNC_PID=""
# 與 drill 相同：crop 幀是中間產物，同步回去要多花 1–2 小時、624MB（09-01 實測）
SYNC_SKIP=(--exclude 'run/offline_two_pass/frames_*/**' --exclude 'run/offline_two_pass/frames/**')

die() { echo "FATAL: $*" >&2; exit 1; }

case "$PROFILE" in
  rankB_deliver_v078|rankB_deliver_v090) ;;
  *) die "未知 DELIVER_PROFILE=$PROFILE" ;;
esac

# ── GPU 閘門（08-31 壞卡事故：ECC 錯誤讓 track_t1 半途 SIGABRT，極易誤判成程式問題）──
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
ECC="$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader | head -1 | tr -d ' ')"
case "$ECC" in
  0|"[N/A]"|"[Not Supported]") ;;
  *) die "GPU ECC uncorrected=$ECC（壞卡，terminate 換機）" ;;
esac
# 09-04 稽核 G20：A100 vs A10 的 SAM3 腿逐列相同僅 79.76%，位元級判準只對 A10 成立。
case "$GPU_NAME" in
  *A10*) ;;
  *) echo "⚠️ WARN：GPU=$GPU_NAME 不是 A10。可以跑，但與 A10 產的參照檔不會逐列相同（G20）。" ;;
esac
echo "GPU=$GPU_NAME ECC=$ECC PROFILE=$PROFILE"

sudo test -x /root/rearm_selfkill.sh || die "缺 /root/rearm_selfkill.sh：cloud-init selfkill 未就位"
sudo /root/rearm_selfkill.sh "${REARM_MIN:-400}"

test -d "$SRC" || die "repo 不在 $REPO"
test -x "$PY1" || die "缺 t1env（先跑 drill 的階段 0 或重建環境）"
test -x "$PY3" || die "缺 sam3env"
test -s "$CKPT/sam3.pt" || die "缺 sam3.pt"
test -s "$CKPT/sam2.1_hiera_large.pt" || die "缺 sam2.1_hiera_large.pt"
test -d "$SAMURAI/sam2" || die "缺 SAMURAI clone"
test -d "$FRAMES" || die "缺 frames-root $FRAMES"
test -s "$SAMPLE" || die "缺 sample $SAMPLE"

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
  echo "$(date -Is) resume finish rc=$RUN_RC" | tee -a "$WORK/finish.txt"
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

mkdir -p "$WORK/logs"
chmod 600 "$RCLONE_CONFIG"
echo "$(date -Is) RESUME_START profile=$PROFILE gpu=$GPU_NAME" | tee -a "$WORK/timeline.txt"

# ── 關鍵步驟：把上一台機器已完成的 run/ 拉回來 ────────────────────────────
# 沒有這一步，run_ranking_b.py 會看到空的 work-dir 而從零開始，且不會有任何警告。
# crop 幀（frames_*）當初刻意沒同步上去，缺了會讓 crop 階段重跑——這是預期的，
# 因為那一階段本來就比較短，而同步它要多花 1–2 小時。
echo "$(date -Is) PULL_RUN_START" | tee -a "$WORK/timeline.txt"
rclone copy "$DEST" "$WORK" --transfers 8 --checkers 16 \
  --exclude '*.part' --exclude '*.part/**' 2>&1 | tail -5
test -d "$WORK/run" || die "gDrive 上沒有 run/——這不是續跑情境，請直接跑 drill"
echo "$(date -Is) PULL_RUN_OK 已完成序列數=$(find "$WORK/run" -name 'submission.csv' | wc -l)" \
  | tee -a "$WORK/timeline.txt"

sync_progress & SYNC_PID=$!

# ── 續跑：旗標與 drill 的階段 2 逐字相同，只多 --resume-existing-work ──────
echo "$(date -Is) RESUME_INFERENCE_START" | tee -a "$WORK/timeline.txt"
"$PY1" "$SRC/run_ranking_b.py" \
  --profile "$PROFILE" --allow-offline-two-pass \
  --frames-root "$FRAMES" --sample "$SAMPLE" \
  --work-dir "$WORK/run" --out "$WORK/final.csv" \
  --sam3-python "$PY3" --samurai-python "$PY1" \
  --sam3-ckpt "$CKPT/sam3.pt" \
  --samurai-dir "$SAMURAI" --samurai-ckpt "$CKPT/sam2.1_hiera_large.pt" \
  --sam3-source-revision "$SAM3_SHA" --samurai-source-revision "$SAMURAI_SHA" \
  --resume-existing-work \
  --execute 2>&1 | tee "$WORK/logs/resume_execute.log"
test -s "$WORK/final.csv" || die "續跑結束但沒有 final.csv"
echo "$(date -Is) RESUME_INFERENCE_DONE" | tee -a "$WORK/timeline.txt"

# final.csv 的 exact-ID-set／unique／finite／domain 驗證由 finalize_submission 在鏈內
# 就做完了（fail-closed，不通過不會產檔）⇒ 這裡不再重複驗，只記錄規模供人工核對。
echo "$(date -Is) RESUME_DONE rows=$(( $(wc -l < "$WORK/final.csv") - 1 ))" | tee -a "$WORK/timeline.txt"
