#!/usr/bin/env bash
# G3（D090 授權鏈最後一關）：把 G2 訓練出的 tracker 權重跑完整 test75，產 LB 候選。
#
# 前提：本腳本設計成【在 G2 那台機器上接著跑】——權重、sam3env、官方 ckpt 都已在本地，
#       只缺 test75 假色資料與 SAMURAI 環境。若在新機器上跑，先跑 G2 腳本的環境段。
#
# 紀律（D090）：訓練產物只能以「新 leg」進 selector 仲裁、LB 單發驗證通過才採用；
#       **絕不 in-place 替換交付管線權重**。本腳本產出的是候選 CSV，不動任何 profile。
#
# ⏰ 自毀：沿用機器上既有的 cloud-init selfkill（唯一計時器）。**執行前先自行 rearm**
#    ——本腳本不動計時器，因為它是接著跑的第二段工作，延多久由操作者依剩餘工作判斷。
set -euo pipefail
trap 'echo "死於第 $LINENO 行" >&2' ERR

export PYTHONUNBUFFERED=1
export PATH="/home/ubuntu/.local/bin:$PATH"
export RCLONE_CONFIG=/home/ubuntu/rclone.conf

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DEST="$GDRIVE/5_outputs/g3_trained_leg_test75_20260829"
REPO=/home/ubuntu/hsot_repo          # ⚠️ 不可叫 hsot（D093：與套件同名撞 namespace）
SRC="$REPO/3_src"
WORK=/home/ubuntu/g3_trained_leg
FRAMES=/home/ubuntu/test_fc
CKPT=/home/ubuntu/ckpt
SAMURAI=/home/ubuntu/samurai
T1ENV=/home/ubuntu/t1env
SAM3ENV=/home/ubuntu/sam3env
PY1="$T1ENV/bin/python"
PY3="$SAM3ENV/bin/python"
SAMURAI_SHA=76ba195984892b0d1e3db5d9c9f90bb62175680a
SAM3_CKPT_SHA256=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
# G2 產物（同機路徑；若換機器則從 gDrive 拉）
G2_BEST="${G2_BEST:-/home/ubuntu/g2_tracker_train/run/tracker_g2_best.pt}"
MERGED="$CKPT/sam3_g2merged.pt"
SYNC_PID=""

die() { echo "FATAL: $*" >&2; exit 1; }
: "${HF_TOKEN:?HF_TOKEN required}"
test -f "$G2_BEST" || die "找不到 G2 best 權重：$G2_BEST"
test -d "$SRC" || die "repo 不在 $REPO"

sync_progress() {
  while true; do
    rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 --exclude '*.part' || true
    sleep 60
  done
}
finish() {
  RC=$?
  trap - EXIT; set +e
  [ -n "$SYNC_PID" ] && { kill "$SYNC_PID" 2>/dev/null; wait "$SYNC_PID" 2>/dev/null; }
  echo "$(date -Is) finish rc=$RC" | tee -a "$WORK/finish.txt"
  rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 --exclude '*.part'
  rclone check "$WORK" "$DEST" --one-way --exclude '*.part'
  exit "$RC"
}
trap finish EXIT

mkdir -p "$WORK/logs" "$FRAMES" "$CKPT"
echo "$(date -Is) G3_START" | tee "$WORK/timeline.txt"

# ── 階段 1：合併權重（tracker.* 覆蓋官方 ckpt，非 tracker 鍵位元級不動）────
test -f "$CKPT/sam3.pt" || die "官方 sam3.pt 不在 $CKPT（同機延跑時應已存在）"
echo "$SAM3_CKPT_SHA256  $CKPT/sam3.pt" | sha256sum -c -
"$PY3" "$SRC/train_tracker_g1/merge_ckpt.py" \
  --base "$CKPT/sam3.pt" --tracker-ckpt "$G2_BEST" --out "$MERGED" \
  2>&1 | tee "$WORK/logs/merge.log"
test -s "$MERGED"
sha256sum "$MERGED" "$G2_BEST" > "$WORK/checkpoint_sha256.txt"
echo "$(date -Is) MERGE_OK" | tee -a "$WORK/timeline.txt"

# ── 階段 2：補齊 test75 資料與 SAMURAI 環境（G2 機器上沒有）──────────────
(
  set -euo pipefail
  if [ "$(find "$FRAMES" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)" -ne 75 ]; then
    rclone copyto "$GDRIVE/1_data/packed/t1test_fc_75.tar" /home/ubuntu/test_fc_75.tar
    tar -xf /home/ubuntu/test_fc_75.tar -C "$FRAMES"
  fi
  test "$(find "$FRAMES" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 75
  rclone copyto "$GDRIVE/1_data/raw/sample_submisson.csv" /home/ubuntu/sample_submisson.csv
  rclone copyto "$GDRIVE/4_models/pretrained/sam2.1_hiera_large.pt" "$CKPT/sam2.1_hiera_large.pt"
  test "$(stat -c%s "$CKPT/sam2.1_hiera_large.pt")" -gt 800000000
  # 組裝需要的既有腿（單變因：source 與第三腿都沿用既有，不重跑）
  mkdir -p "$REPO/5_outputs/submissions"
  rclone copy "$GDRIVE/5_outputs/submissions" "$REPO/5_outputs/submissions" \
    --include 'sub_v012_*.csv' --include 'sub_v078_*.csv' --transfers 8
  rclone copy "$GDRIVE/5_outputs/rankb_robust_test75_20260822/run/full_sam3" \
    "$REPO/5_outputs/rankb_robust_test75_20260822/run/full_sam3" --include 'submission.csv'
  echo DATA_READY
) > "$WORK/logs/data.log" 2>&1 &
DATA_PID=$!

[ -x "$T1ENV/bin/python" ] || uv venv --python 3.12 "$T1ENV"
[ -d "$SAMURAI/.git" ] || git clone https://github.com/yangchris11/samurai.git "$SAMURAI"
git -C "$SAMURAI" fetch --depth 1 origin "$SAMURAI_SHA"
git -C "$SAMURAI" checkout --detach "$SAMURAI_SHA"
VIRTUAL_ENV="$T1ENV" uv pip install -q torch torchvision --torch-backend=auto
VIRTUAL_ENV="$T1ENV" uv pip install -q -e "$SAMURAI/sam2" \
  scipy loguru tqdm pandas pillow opencv-python-headless
wait "$DATA_PID"; grep -q DATA_READY "$WORK/logs/data.log"
echo "$(date -Is) DATA_ENV_OK" | tee -a "$WORK/timeline.txt"

sync_progress & SYNC_PID=$!

# ── 階段 3：走既有單一入口，唯一差別＝ --sam3-ckpt 指向合併權重 ──────────
# 兩腿都跑（run_ranking_b 單一 frames-root 架構）；SAMURAI 用原權重不受影響。
# 組裝的單變因處理在階段 4。
echo "$(date -Is) INFERENCE_START" | tee -a "$WORK/timeline.txt"
"$PY1" "$SRC/run_ranking_b.py" \
  --profile rankB_deliver_v078 --allow-offline-two-pass \
  --frames-root "$FRAMES" --sample /home/ubuntu/sample_submisson.csv \
  --work-dir "$WORK/run" --out "$WORK/g3_full.csv" \
  --sam3-python "$PY3" --samurai-python "$PY1" \
  --sam3-ckpt "$MERGED" \
  --samurai-dir "$SAMURAI" --samurai-ckpt "$CKPT/sam2.1_hiera_large.pt" \
  --execute 2>&1 | tee "$WORK/logs/execute.log"
test -s "$WORK/g3_full.csv"
echo "$(date -Is) INFERENCE_DONE" | tee -a "$WORK/timeline.txt"

# ── 階段 4：單變因候選（trained SAM3 腿 ＋ 既有 SAMURAI／第三腿）────────
# 為何：SAMURAI 腿在本次是用原權重重跑的，但 D074 已實測跨執行約 13% 序列會發散
# ⇒ 若採本次的 SAMURAI 輸出，變因就不只是「tracker 權重」。改用既有 sub_v012，
# 讓與 v078 的唯一差異就是主線腿的權重。
"$PY1" "$SRC/finalize_submission.py" \
  --main "$WORK/run/offline_two_pass/main_merged.csv" \
  --source "$REPO/5_outputs/submissions/sub_v012_e23b_sam21_ablation.csv" \
  --sample /home/ubuntu/sample_submisson.csv \
  --corr both --K 6 --selector v2 --qhead v056 \
  --third-leg "$REPO/5_outputs/rankb_robust_test75_20260822/run/full_sam3/submission.csv" \
  --out "$WORK/g3_singlevar.csv" 2>&1 | tee "$WORK/logs/finalize_singlevar.log"
test -s "$WORK/g3_singlevar.csv"

# ── 階段 5：與 v078 的逐序列歸因（無 GT，量的是改了多少、哪裡改）──────────
"$PY1" - "$WORK/g3_singlevar.csv" "$REPO/5_outputs/submissions/sub_v078_thirdleg_deadzone.csv" \
  > "$WORK/g3_vs_v078.txt" 2>&1 <<'PY' || true
import csv, sys
def load(p):
    with open(p, newline="") as fh:
        return {r["ID"]: tuple(float(r[c]) for c in ("x","y","width","height"))
                for r in csv.DictReader(fh)}
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
print(f"共同列 {len(common)}；逐列相同 {same} ({100*same/len(common):.1f}%)")
print(f"對 v078 平均 IoU {sum(sum(v)/len(v) for v in seqs.values())/len(seqs):.5f}")
print("改動最大的 10 支序列（IoU 越低＝改越多）:")
for m, s in sorted((sum(v)/len(v), s) for s, v in seqs.items())[:10]:
    print(f"  {s}: {m:.4f}")
PY
cat "$WORK/g3_vs_v078.txt"
echo "$(date -Is) G3_DONE" | tee -a "$WORK/timeline.txt"
