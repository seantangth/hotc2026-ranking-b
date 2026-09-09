#!/usr/bin/env bash
# D097 run F：把兩個 venv 釘死回 08-22 的套件版本後，重跑同一條 rankB_deliver_v078 鏈。
#
# 為什麼要有這一發（08-30 發現）：
#   D（08-30 本機）與演練 #4（08-29 另一台機器）產出**位元級完全相同** ⇒ 這條鏈在
#   相同套件版本下是**確定性的**，「執行抽籤」不存在。因此 v083（0.70701）與 v078
#   （0.71666）的 0.00965 落差**不是方差，是系統性差異**（0 支序列完全相同）。
#   已定位：同一條 full-SAM3 腿在 08-22 環境 vs 08-29/30 環境下輸出不同（第 12 行就分歧）。
#   兩個 env 的實際差異（逐檔 diff，非 grep 混檔）：
#     sam3env: timm 1.0.28→1.0.29、cuda-pathfinder 1.6.1→1.8.0、
#              click／filelock／huggingface-hub／portalocker／wcwidth（工具類）
#     t1env:   hydra-core 1.3.5→1.3.6、cuda-pathfinder、filelock、portalocker
#   ⚠️ opencv 與 numpy **兩邊相同**（先前「opencv 4.11→5.0」的說法是 grep 跨檔造成的假象）。
#
# 本腳本假設 venv 已就地降版完成（本次由互動式指令完成並逐項 diff 驗證：
# sam3env 與 08-22 完全一致；t1env 僅多 pytest 工具，不進推論路徑）。
# 它只做「跑一次 ＋ 存證 ＋ 同步 ＋ 自毀」。
set -euo pipefail
trap 'echo "死於第 $LINENO 行" >&2' ERR

export PYTHONUNBUFFERED=1
export PATH="/home/ubuntu/.local/bin:$PATH"
export RCLONE_CONFIG=/home/ubuntu/rclone.conf

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DEST="$GDRIVE/5_outputs/consensus_2runs_20260830"
REPO=/home/ubuntu/hsot_repo
SRC="$REPO/3_src"
WORK=/home/ubuntu/consensus_2runs
FRAMES=/home/ubuntu/test_fc
CKPT=/home/ubuntu/ckpt
SAMURAI=/home/ubuntu/samurai
PY1=/home/ubuntu/t1env/bin/python
PY3=/home/ubuntu/sam3env/bin/python
SAMURAI_SHA=76ba195984892b0d1e3db5d9c9f90bb62175680a
SAM3_SHA=96914d2425f90a64f45ca977c2b5165418099543
SAMPLE=/home/ubuntu/sample_submisson.csv

die() { echo "FATAL: $*" >&2; exit 1; }

finish() {
  RUN_RC=$?
  trap - EXIT
  set +e
  echo "$(date -Is) F finish rc=$RUN_RC" | tee -a "$WORK/finish_F.txt"
  rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 \
    --exclude '*.part' --exclude '*.part/**'
  SYNC_RC=$?
  [ "$SYNC_RC" -eq 0 ] && rclone check "$WORK" "$DEST" --one-way \
    --exclude '*.part' --exclude '*.part/**' || SYNC_RC=$?
  if [ "$SYNC_RC" -eq 0 ]; then
    sudo /root/rearm_selfkill.sh 5
  else
    echo "$(date -Is) F REMOTE_VERIFY_FAILED rc=$SYNC_RC；selfkill 未縮短" | tee -a "$WORK/finish_F.txt"
  fi
  exit "$RUN_RC"
}
trap finish EXIT

# 存證：這一發實際用的套件版本（判決書要引用）
uv pip freeze --python "$PY1" > "$WORK/t1_freeze_F_pinned.txt"
uv pip freeze --python "$PY3" > "$WORK/sam3_freeze_F_pinned.txt"
"$PY3" -c "import timm,numpy; assert timm.__version__=='1.0.28', timm.__version__; assert numpy.__version__=='1.26.4', numpy.__version__" \
  || die "sam3env 未釘死到 08-22 版本"
"$PY1" -c "import hydra; assert hydra.__version__=='1.3.5', hydra.__version__" \
  || die "t1env 的 hydra-core 未釘死到 1.3.5"

echo "$(date -Is) F_INFERENCE_START" | tee -a "$WORK/timeline.txt"
"$PY1" "$SRC/run_ranking_b.py" \
  --profile rankB_deliver_v078 --allow-offline-two-pass \
  --frames-root "$FRAMES" --sample "$SAMPLE" \
  --work-dir "$WORK/run_F" --out "$WORK/final_F.csv" \
  --sam3-python "$PY3" --samurai-python "$PY1" \
  --sam3-ckpt "$CKPT/sam3.pt" \
  --samurai-dir "$SAMURAI" --samurai-ckpt "$CKPT/sam2.1_hiera_large.pt" \
  --sam3-source-revision "$SAM3_SHA" --samurai-source-revision "$SAMURAI_SHA" \
  --execute 2>&1 | tee "$WORK/logs/execute_F.log"
test -s "$WORK/final_F.csv"
echo "$(date -Is) F_INFERENCE_DONE" | tee -a "$WORK/timeline.txt"

# 三方比對：F vs D（＝純環境效應）、F vs sub_v078（＝離目標多遠）
"$PY1" - "$WORK/final_F.csv" "$WORK/final_D.csv" "$REPO/5_outputs/submissions/sub_v078_thirdleg_deadzone.csv" \
  > "$WORK/compare_F.txt" 2>&1 <<'PY' || true
import csv, sys
def load(p):
    with open(p, newline="") as fh:
        return {r["ID"]: tuple(float(r[c]) for c in ("x","y","width","height"))
                for r in csv.DictReader(fh)}
def iou(a, b):
    iw = max(0.0, min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1]))
    inter = iw*ih; union = a[2]*a[3] + b[2]*b[3] - inter
    return inter/union if union > 0 else 0.0
F, D, A = load(sys.argv[1]), load(sys.argv[2]), load(sys.argv[3])
for tag, ref in (("F vs D (純環境效應)", D), ("F vs sub_v078 (離目標)", A)):
    common = set(F) & set(ref)
    same = sum(1 for k in common if F[k] == ref[k])
    seqs = {}
    for k in common:
        seqs.setdefault(k.rsplit("_",1)[0], []).append(iou(F[k], ref[k]))
    means = {s: sum(v)/len(v) for s, v in seqs.items()}
    print(f"--- {tag} ---")
    print(f"逐列相同 {same}/{len(common)} ({100*same/max(len(common),1):.2f}%)")
    print(f"序列平均 IoU {sum(means.values())/max(len(means),1):.5f}")
    print(f"完全相同的序列 {sum(1 for m in means.values() if m == 1.0)} 支／發散(<0.5) {sum(1 for m in means.values() if m < 0.5)} 支")
    for s, m in sorted(means.items(), key=lambda kv: kv[1])[:5]:
        print(f"   {s}: {m:.4f}")
PY
cat "$WORK/compare_F.txt"
echo "$(date -Is) F_DONE" | tee -a "$WORK/timeline.txt"
exit 0
