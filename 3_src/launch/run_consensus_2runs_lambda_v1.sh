#!/usr/bin/env bash
# D096 對等三執行共識驗證：同一台機器連續跑 2 次【完全相同】的 rankB_deliver_v078 鏈。
#
# 目的：08-30 量到重跑方差 0.00965（v078 0.71666 / v081 0.71246 / v083 0.70701），
#       大於剩餘所有加分手段總和。本次產出兩份對等執行 D、E，連同現有的 A（sub_v078）
#       湊成【全 75 支都有三份獨立答案】的三方共識，驗證 9/7 該不該跑三次取 medoid。
#
# ⚠️ 為何是 v078 鏈而不是 v082 鏈：v082 鏈沒有單一 profile（第四腿只活在
#    finalize_submission --fourth-leg 的獨立 CPU 步驟），且現任 A 本身就是 v078 鏈
#    ——混用會讓三次執行再度不對等，正是 v084 栽的坑。第四腿是正交的 CPU 加法軸。
#
# ⚠️ 為何同機而非兩台並行：9/7 的部署情境就是同一台機器連續跑。同機方差是保守下界
#    （跨機還多了環境差異），同機若已足以讓共識生效，跨機只會更有效。
#
# 本腳本改自 run_deliver_v078_drill_lambda_v1.sh（演練 #4 rc=0 驗證通過），
# 差異僅四處：WORK/DEST、自毀分鐘、階段 2 跑兩次、階段 3 改為三方比較＋產 medoid。
# 環境建置與階段 0/1 閘門【逐字照抄】，不重新發明。
#
# ⏰ 自毀：唯一計時器＝cloud-init /root/rearm_selfkill.sh（開機端武裝）。
#    本腳本【不】自掛第二顆 sleep（08-07 事故：兩套計時器只延了一套）。
#    開工 rearm 420 分＝環境 1.7h ＋ 推論 2×1.7h ＋ 收尾緩衝 ×2。
set -euo pipefail
trap 'echo "死於第 $LINENO 行" >&2' ERR

export PYTHONUNBUFFERED=1
export PATH="/home/ubuntu/.local/bin:$PATH"
export RCLONE_CONFIG=/home/ubuntu/rclone.conf

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DEST="$GDRIVE/5_outputs/consensus_2runs_20260830"
REPO=/home/ubuntu/hsot_repo          # D093：不可叫 hsot（與套件同名撞 namespace）
SRC="$REPO/3_src"
WORK=/home/ubuntu/consensus_2runs
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
SYNC_PID=""

die() { echo "FATAL: $*" >&2; exit 1; }

: "${HF_TOKEN:?HF_TOKEN is required (gated facebook/sam3); export it before running}"
sudo test -x /root/rearm_selfkill.sh \
  || die "缺 /root/rearm_selfkill.sh：cloud-init selfkill backstop 未就位（本腳本不自掛計時器）"
sudo /root/rearm_selfkill.sh 420
test -d "$SRC" || die "repo 不在 $REPO（開機流程須把 3_src 放這裡）"

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
  [ -n "$SYNC_PID" ] && { kill "$SYNC_PID" 2>/dev/null; wait "$SYNC_PID" 2>/dev/null; }
  echo "$(date -Is) finish rc=$RUN_RC" | tee -a "$WORK/finish.txt"
  SYNC_RC=0
  if command -v rclone >/dev/null 2>&1 && [ -f "$RCLONE_CONFIG" ]; then
    rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 \
      --exclude '*.part' --exclude '*.part/**' || SYNC_RC=$?
    [ "$SYNC_RC" -eq 0 ] && { rclone check "$WORK" "$DEST" --one-way \
      --exclude '*.part' --exclude '*.part/**' || SYNC_RC=$?; }
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
echo "$(date -Is) SETUP_START" | tee "$WORK/timeline.txt"

command -v rclone >/dev/null 2>&1 || curl -fsSL https://rclone.org/install.sh | sudo bash >/dev/null
rclone lsf "$GDRIVE/1_data/packed" >/dev/null
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null

# ── 資料／權重（背景並行；D036：單一 tar，不逐檔 rclone）────────────────────
(
  set -euo pipefail
  rclone copyto "$GDRIVE/1_data/packed/t1test_fc_75.tar" /home/ubuntu/test_fc_75.tar
  tar -xf /home/ubuntu/test_fc_75.tar -C "$FRAMES"
  test "$(find "$FRAMES" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 75
  rclone copyto "$GDRIVE/1_data/raw/sample_submisson.csv" "$SAMPLE"
  test -s "$SAMPLE"
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
  curl -fL -H "Authorization: Bearer $HF_TOKEN" \
    -o "$CKPT/sam3.pt" "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt"
  echo "$SAM3_CKPT_SHA256  $CKPT/sam3.pt" | sha256sum -c -
  echo "official gated: https://huggingface.co/facebook/sam3/resolve/main/sam3.pt (sha256 $SAM3_CKPT_SHA256)" \
    > "$WORK/SAM3_WEIGHT_SOURCE.txt"
  echo SAM3_READY
) > "$WORK/logs/sam3_download.log" 2>&1 &
SAM3_PID=$!

# ── 兩個完全隔離的 venv（禁 --system-site-packages）───────────────────────
[ -x "$T1ENV/bin/python" ] || uv venv --python 3.12 "$T1ENV"
[ -d "$SAMURAI/.git" ] || git clone https://github.com/yangchris11/samurai.git "$SAMURAI"
git -C "$SAMURAI" fetch --depth 1 origin "$SAMURAI_SHA"
git -C "$SAMURAI" checkout --detach "$SAMURAI_SHA"
VIRTUAL_ENV="$T1ENV" uv pip install -q torch torchvision --torch-backend=auto
VIRTUAL_ENV="$T1ENV" uv pip install -q -e "$SAMURAI/sam2" \
  scipy loguru tqdm pandas pillow opencv-python-headless pytest
"$PY1" -c "import torch,sam2,pandas; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

[ -x "$SAM3ENV/bin/python" ] || uv venv --python 3.12 "$SAM3ENV"
VIRTUAL_ENV="$SAM3ENV" uv pip install -q torch torchvision --torch-backend=auto
VIRTUAL_ENV="$SAM3ENV" uv pip install -q \
  "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}" \
  "numpy<2" ftfy==6.1.1 regex einops psutil scipy av pycocotools numba \
  python-rapidjson pandas pillow tqdm opencv-python-headless
VIRTUAL_ENV="$SAM3ENV" uv pip install -q "setuptools<81"   # D086：81+ 移除 pkg_resources
"$PY3" -c "import torch,sam3,numpy; assert torch.cuda.is_available(); assert numpy.__version__.startswith('1.'); print(torch.__version__, torch.cuda.get_device_name(0))"

for JOB_PID in "$DATA_PID" "$SAM21_PID" "$SAM3_PID"; do wait "$JOB_PID"; done
grep -q DATA_READY "$WORK/logs/data_setup.log"
grep -q SAM21_READY "$WORK/logs/sam21_download.log"
grep -q SAM3_READY "$WORK/logs/sam3_download.log"

sha256sum "$CKPT/sam2.1_hiera_large.pt" "$CKPT/sam3.pt" > "$WORK/checkpoint_sha256.txt"
git -C "$SAMURAI" rev-parse HEAD > "$WORK/samurai_commit.txt"
uv pip freeze --python "$PY1" > "$WORK/t1_freeze.txt"
uv pip freeze --python "$PY3" > "$WORK/sam3_freeze.txt"
nvidia-smi > "$WORK/nvidia_smi.txt" || true

sync_progress &
SYNC_PID=$!

# ── 階段 0：交付自測（照抄演練腳本；在燒 3.4h GPU 前先證明鏈仍位元級等於 v078）──
echo "$(date -Is) SELFTEST_START" | tee -a "$WORK/timeline.txt"
mkdir -p "$REPO"/5_outputs/submissions "$REPO"/1_data/raw
rclone copy "$GDRIVE/5_outputs/submissions" "$REPO"/5_outputs/submissions \
  --include 'sub_v023_*.csv' --include 'sub_v012_*.csv' --include 'sub_v049_*.csv' \
  --include 'sub_v056_*.csv' --include 'sub_v078_*.csv' --transfers 8
rclone copy "$GDRIVE/5_outputs/rankb_robust_test75_20260822/run/full_sam3" \
  "$REPO"/5_outputs/rankb_robust_test75_20260822/run/full_sam3 \
  --include 'submission.csv' --transfers 4
cp "$SAMPLE" "$REPO"/1_data/raw/sample_submisson.csv
"$PY1" "$SRC/finalize_submission.py" --selftest 2>&1 | tee "$WORK/logs/selftest.log"
grep -q "自測 (c) 通過" "$WORK/logs/selftest.log" \
  || die "selftest (c) 未通過或被略過——交付鏈未經位元級驗證，停工"
(cd "$REPO" && PYTHONPATH="$SRC" "$PY1" -m pytest -q 3_src \
   --ignore=3_src/hsot/test_e32b_pilot.py) 2>&1 | tee "$WORK/logs/pytest.log"
echo "$(date -Is) SELFTEST_OK" | tee -a "$WORK/timeline.txt"

# ── 階段 1：dry-run（BLOCK=0 才准 execute）────────────────────────────────
echo "$(date -Is) DRYRUN_START" | tee -a "$WORK/timeline.txt"
"$PY1" "$SRC/run_ranking_b.py" \
  --profile rankB_deliver_v078 --allow-offline-two-pass \
  --frames-root "$FRAMES" --sample "$SAMPLE" \
  --work-dir "$WORK/run_D" --out "$WORK/final_D.csv" \
  --sam3-python "$PY3" --samurai-python "$PY1" \
  --sam3-ckpt "$CKPT/sam3.pt" \
  --samurai-dir "$SAMURAI" --samurai-ckpt "$CKPT/sam2.1_hiera_large.pt" \
  --sam3-source-revision "$SAM3_SHA" --samurai-source-revision "$SAMURAI_SHA" \
  2>&1 | tee "$WORK/logs/dryrun.log"
grep -q "BLOCK=0" "$WORK/logs/dryrun.log" || die "dry-run 有 BLOCK，停工（見 logs/dryrun.log）"

# ── 階段 2：兩次對等執行 ──────────────────────────────────────────────────
# ⚠️ work-dir 與 out 必須各自全新：run_ranking_b.py 的 output-collision 檢查在
#    execute 時對「work 非空 或 out 已存在」判 BLOCK（見 run_ranking_b.py:558-568）。
#    這道檢查同時也是本實驗的護欄——它保證第二次是真的重跑，不是重用第一次的產物。
for RUN_ID in D E; do
  echo "$(date -Is) INFERENCE_${RUN_ID}_START" | tee -a "$WORK/timeline.txt"
  "$PY1" "$SRC/run_ranking_b.py" \
    --profile rankB_deliver_v078 --allow-offline-two-pass \
    --frames-root "$FRAMES" --sample "$SAMPLE" \
    --work-dir "$WORK/run_$RUN_ID" --out "$WORK/final_${RUN_ID}.csv" \
    --sam3-python "$PY3" --samurai-python "$PY1" \
    --sam3-ckpt "$CKPT/sam3.pt" \
    --samurai-dir "$SAMURAI" --samurai-ckpt "$CKPT/sam2.1_hiera_large.pt" \
    --sam3-source-revision "$SAM3_SHA" --samurai-source-revision "$SAMURAI_SHA" \
    --execute 2>&1 | tee "$WORK/logs/execute_${RUN_ID}.log"
  test -s "$WORK/final_${RUN_ID}.csv"
  echo "$(date -Is) INFERENCE_${RUN_ID}_DONE" | tee -a "$WORK/timeline.txt"
done

# ── 階段 3：三方 medoid 共識 ＋ 逐兩兩比較 ────────────────────────────────
# A ＝ sub_v078（LB 0.71666，已在 repo 內）、D／E ＝ 本次兩份對等執行。
# 工具＝3_src/run_ensemble_medoid.py（本機已驗證能位元級重現 08-30 的 sub_v084）。
echo "$(date -Is) CONSENSUS_START" | tee -a "$WORK/timeline.txt"
A_CSV="$REPO/5_outputs/submissions/sub_v078_thirdleg_deadzone.csv"
"$PY1" "$SRC/run_ensemble_medoid.py" \
  "$A_CSV" "$WORK/final_D.csv" "$WORK/final_E.csv" \
  --out "$WORK/sub_consensus_ADE.csv" --report "$WORK/consensus_report.txt" \
  2>&1 | tee "$WORK/logs/consensus.log"
test -s "$WORK/sub_consensus_ADE.csv"

# 兩兩比較：D vs A、E vs A、D vs E ⇒ 量「同機重跑」的方差形狀（v083 是跨機器的）
for PAIR in "D:$WORK/final_D.csv:$A_CSV" "E:$WORK/final_E.csv:$A_CSV" \
            "DE:$WORK/final_D.csv:$WORK/final_E.csv"; do
  TAG="${PAIR%%:*}"; REST="${PAIR#*:}"; LHS="${REST%%:*}"; RHS="${REST#*:}"
  "$PY1" - "$LHS" "$RHS" > "$WORK/compare_${TAG}.txt" 2>&1 <<'PY' || true
import csv, sys
def load(p):
    with open(p, newline="") as fh:
        return {r["ID"]: tuple(float(r[c]) for c in ("x","y","width","height"))
                for r in csv.DictReader(fh)}
got, ref = load(sys.argv[1]), load(sys.argv[2])
common = set(got) & set(ref)
same = sum(1 for k in common if got[k] == ref[k])
def iou(a, b):
    iw = max(0.0, min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1]))
    inter = iw*ih; union = a[2]*a[3] + b[2]*b[3] - inter
    return inter/union if union > 0 else 0.0
seqs = {}
for k in common:
    seqs.setdefault(k.rsplit("_",1)[0], []).append(iou(got[k], ref[k]))
means = {s: sum(v)/len(v) for s, v in seqs.items()}
print(f"common={len(common)}  逐列相同={same} ({100*same/max(len(common),1):.2f}%)")
print(f"平均 IoU={sum(means.values())/max(len(means),1):.5f}")
print(f"發散序列(<0.5)={sum(1 for m in means.values() if m < 0.5)} 支")
for s, m in sorted(means.items(), key=lambda kv: kv[1])[:5]:
    print(f"  {s}: {m:.4f}")
PY
  echo "--- compare $TAG ---"; cat "$WORK/compare_${TAG}.txt"
done
echo "$(date -Is) CONSENSUS_DONE" | tee -a "$WORK/timeline.txt"
exit 0
