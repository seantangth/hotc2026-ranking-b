#!/usr/bin/env bash
# ============================================================================
# setup_e31_protocol_probe_v1.sh — E31 訓練協定對齊 micro-probe（兩腿 LR）
#
# 設計與事前判準：5_outputs/e31_protocol_probe_20260812/DESIGN.md（執行前寫死）
# 目的：否證/驗證 D065(e) 假說①（train/eval 協定不匹配）＝ D065 要求的獨立實驗
#
# 【與 Phase 1 的唯一差異】協定對齊（make_train_cfg_v2 --align-protocol）＋ LR 兩腿
# 【不做】G1 / diag19 / 推論 / LB —— 本探針只測「目標函數可否被優化」
#
# 🚨 terminate 一律用「本腳本收到的 INSTANCE_ID」，**不用 name 匹配**
#    （Sean 08-12 指示：只能關自己這個 session 開的機器）
# 【紀律】D016 每階段 rclone｜D018 用完 terminate｜計時器盤點（08-07 教訓）
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

INSTANCE_ID="${INSTANCE_ID:?必須傳入 INSTANCE_ID（terminate 用它，不用 name）}"
GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DEST="$GDRIVE/5_outputs/e31_protocol_probe_20260812"
SAM2REPO=~/sam2repo
SAM2_SHA=2b90b9f5ceec907a1c18123530e92e794ad901a4   # 與 Phase 1 同 SHA（D034 釘死）
PYT=~/trainenv/bin/python
EPOCHS="${EPOCHS:-8}"          # 8×152 = 1,216 步/腿（判準分位需要足夠樣本）
OUT=~/e31_out; LOGS=~/e31_logs
STAMP=~/e31_timing.txt; : > "$STAMP"
mkdir -p "$OUT" "$LOGS"

selfkill() {   # 用 id 精準終止，絕不誤傷他人機器
  key=$(cat ~/.lambda_key | tr -d '\n'); [ -z "$key" ] && return 0
  curl -s -u "$key:" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H 'Content-Type: application/json' -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}"
}
die() {
  echo "🚨 $*"
  rclone copy "$OUT" "$DEST/out" --transfers 8 2>/dev/null
  rclone copy "$LOGS" "$DEST/logs" --include "*.log" --transfers 8 2>/dev/null
  echo "DIED: $*" > /tmp/e31_died.txt
  rclone copyto /tmp/e31_died.txt "$DEST/DIED.txt" 2>/dev/null
  selfkill; exit 1
}
mark() { echo "$1 $(date +%s)" >> "$STAMP"; echo "⏱  $1 @ $(date +%H:%M:%S)"; }
mark E31_START

# --- [0] 保險自毀（用 id）------------------------------------------------------
echo "=== [0] 計時器盤點 ==="
pgrep -af "sleep [0-9]+" || echo "（無既有計時器）"
nohup bash -c "sleep 5400
  key=\$(cat ~/.lambda_key | tr -d '\n'); [ -z \"\$key\" ] && exit 0
  curl -s -u \"\$key:\" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H 'Content-Type: application/json' -d '{\"instance_ids\":[\"$INSTANCE_ID\"]}'
" > ~/e31_insurance.log 2>&1 &
echo "⏰ 90 分鐘保險自毀已掛（id=$INSTANCE_ID）"

# --- [0.5] rclone（全新機器沒有；Phase 1 是靠 drill 預先裝好，本腳本不能假設）---
echo "=== [0.5] rclone 安裝檢查 ==="
if ! command -v rclone >/dev/null; then
  echo "安裝 rclone…"
  curl -s https://rclone.org/install.sh | sudo bash >/dev/null 2>&1 \
    || { sudo apt-get -qq update && sudo apt-get -qq install -y rclone; }
fi
command -v rclone >/dev/null || die "rclone 安裝失敗"
[ -s ~/.config/rclone/rclone.conf ] || die "rclone.conf 未上傳（本機 scp 應已送達）"
rclone lsf "$GDRIVE/" >/dev/null 2>&1 || die "rclone 無法存取 gDrive（設定或授權問題）"
echo "✅ rclone $(rclone version | head -1 | awk '{print $2}') 可用且 gDrive 可存取"

# --- [1] 背景拉資料 -----------------------------------------------------------
echo "=== [1] 背景拉資料（單 tar，D036）==="
( set -e; rclone copyto "$GDRIVE/1_data/packed/train_nir_fc.tar" ~/train_fc.tar
  mkdir -p ~/train_fc && tar -xf ~/train_fc.tar -C ~/train_fc && echo TRAINFC_READY
) > ~/trainfc.log 2>&1 &
TRAINFC_PID=$!
( set -e; mkdir -p ~/ckpt
  rclone copyto "$GDRIVE/4_models/pretrained/sam2.1_hiera_large.pt" ~/ckpt/sam2.1_hiera_large.pt
  echo CKPT_READY ) > ~/ckpt.log 2>&1 &
CKPT_PID=$!

rclone copyto "$GDRIVE/5_outputs/ef_phase1_20260811/davis_nir_ann.tar" ~/davis_nir_ann.tar \
  || die "teacher annotations 拉取失敗"
mkdir -p ~/davis_nir && tar -xf ~/davis_nir_ann.tar -C ~/davis_nir || die "ann tar 解開失敗"
[ -s ~/davis_nir/train_list_pruned.txt ] || die "train_list_pruned.txt 缺"
for f in freeze_spec_v1.py make_train_cfg_v2.py e31_verdict_v1.py; do
  rclone copyto "$GDRIVE/3_src/peft/$f" ~/$f || die "拉 $f 失敗"
  [ -s ~/$f ] || die "$f 空檔"
done
rclone copyto "$GDRIVE/5_outputs/ef_phase1_20260811/logs/stageB_console.log" \
  ~/phase1_stageB.log || echo "⚠️ Phase 1 log 拉取失敗，判準改用 DESIGN 記錄的錨點"

# --- [2] trainenv（完全隔離 venv，CLAUDE.md 鐵律）------------------------------
echo "=== [2] trainenv + 官方 sam2 @ SHA ==="
mark ENV_START
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d "$SAM2REPO" ] || git clone -q https://github.com/facebookresearch/sam2.git "$SAM2REPO"
( cd "$SAM2REPO" && git checkout -q "$SAM2_SHA" ) || die "sam2 SHA checkout 失敗"
echo "sam2repo SHA = $SAM2_SHA" > "$OUT/pins.txt"
[ -d ~/trainenv ] || uv venv --python 3.12 ~/trainenv
if [ ! -f ~/trainenv/.deps_done ]; then
  VIRTUAL_ENV=~/trainenv uv pip install -q torch torchvision --torch-backend=auto || die "torch 失敗"
  ( cd "$SAM2REPO" && VIRTUAL_ENV=~/trainenv uv pip install -q -e ".[dev]" ) || die "sam2[dev] 失敗"
  VIRTUAL_ENV=~/trainenv uv pip install -q submitit tensorboard opencv-python-headless || die "附屬失敗"
  touch ~/trainenv/.deps_done
fi
cp ~/freeze_spec_v1.py "$SAM2REPO/freeze_spec_v1.py"
$PYT - <<'PY' || die "trainenv 驗證失敗"
import torch, sam2, hydra, omegaconf, submitit, os, sys
assert torch.cuda.is_available()
sys.path.insert(0, os.path.expanduser("~/sam2repo"))
import training.train, freeze_spec_v1  # noqa
print(f"trainenv OK: torch {torch.__version__}  gpu={torch.cuda.get_device_name(0)}")
PY
mark ENV_DONE

wait $TRAINFC_PID; grep -q TRAINFC_READY ~/trainfc.log || die "train_fc 拉取失敗"
wait $CKPT_PID;    grep -q CKPT_READY    ~/ckpt.log    || die "權重拉取失敗"

# --- [3] 重建 DAVIS JPEGImages（tar 內只有 Annotations；JPEG 原為 symlink）-----
echo "=== [3] 重建 JPEGImages symlink 樹 + fail-closed 驗證 ==="
mark REBUILD_START
TRAIN_ROOT=~/train_fc
FIRST_SEQ=$(head -1 ~/davis_nir/train_list_pruned.txt)
if [ ! -d "$TRAIN_ROOT/$FIRST_SEQ" ]; then
  CAND=$(find ~/train_fc -maxdepth 2 -type d -name "$FIRST_SEQ" -print -quit)
  [ -n "$CAND" ] || die "train tar 內找不到 $FIRST_SEQ"
  TRAIN_ROOT=$(dirname "$CAND")
fi
echo "TRAIN_ROOT=$TRAIN_ROOT"
TRAIN_ROOT="$TRAIN_ROOT" python3 - <<'PY' || die "JPEGImages 重建失敗"
import os, sys
from pathlib import Path
root = Path(os.environ["TRAIN_ROOT"]); davis = Path.home()/"davis_nir"
seqs = [s for s in (davis/"train_list_pruned.txt").read_text().split() if s]
# 對位規則取自 teacher_davis_v2.py L104/L121-125（唯一真相源）：
#   jpgs = sorted(glob("*.jpg")); dst = f"{pos:05d}.jpg" -> jpgs[pos]
# prune 後 Annotations 只留有標註的幀 ⇒ 依 Annotation 檔名決定要建哪些 symlink
bad = []
for seq in seqs:
    ann_d = davis/"Annotations"/seq
    anns = sorted(ann_d.glob("*.png"))
    if not anns: bad.append(f"{seq}: 無 Annotation"); continue
    jpgs = sorted((root/seq).glob("*.jpg"))
    if not jpgs: bad.append(f"{seq}: 原始影格不存在"); continue
    img_d = davis/"JPEGImages"/seq; img_d.mkdir(parents=True, exist_ok=True)
    for a in anns:
        pos = int(a.stem)
        if pos >= len(jpgs):
            bad.append(f"{seq}: pos {pos} 超出影格數 {len(jpgs)}"); break
        dst = img_d/f"{pos:05d}.jpg"
        if not dst.exists(): dst.symlink_to(jpgs[pos])
if bad:
    print("🚨 重建失敗：", *bad[:10], sep="\n  "); sys.exit(1)
# fail-closed：每支序列 JPEG 與 Annotation 的檔名集合必須完全一致（防 D059 型靜默錯位）
mism = []
for seq in seqs:
    j = {p.stem for p in (davis/"JPEGImages"/seq).glob("*.jpg")}
    a = {p.stem for p in (davis/"Annotations"/seq).glob("*.png")}
    if j != a: mism.append(f"{seq}: jpg {len(j)} vs ann {len(a)}，差集 {sorted(j^a)[:5]}")
    if any(not (davis/"JPEGImages"/seq/f"{s}.jpg").resolve().exists() for s in list(j)[:20]):
        mism.append(f"{seq}: 有斷鏈 symlink")
if mism:
    print("🚨 對位驗證失敗：", *mism[:10], sep="\n  "); sys.exit(1)
tot = sum(len(list((davis/"JPEGImages"/s).glob("*.jpg"))) for s in seqs)
print(f"✅ 重建並驗證通過：{len(seqs)} 支序列、{tot} 幀，JPEG↔Annotation 檔名集合逐支一致")
PY
mark REBUILD_DONE
rclone copy "$OUT" "$DEST/out" --transfers 8

# --- [4] 兩腿訓練 -------------------------------------------------------------
CFG_DIR="$SAM2REPO/sam2/configs/sam2.1_training"
run_leg() {   # $1=腿名  $2=lr
  local leg=$1 lr=$2
  echo "=== [4-$leg] 協定對齊 @ lr=$lr（$EPOCHS epochs）==="
  mark "LEG_${leg}_START"
  $PYT ~/make_train_cfg_v2.py --sam2-repo "$SAM2REPO" \
    --img-folder ~/davis_nir/JPEGImages --gt-folder ~/davis_nir/Annotations \
    --file-list ~/davis_nir/train_list_pruned.txt --ckpt ~/ckpt/sam2.1_hiera_large.pt \
    --num-epochs "$EPOCHS" --log-dir "$LOGS/leg_$leg" --num-workers 8 --save-freq 0 \
    --base-lr "$lr" --align-protocol \
    --out "$CFG_DIR/hsot_e31_leg_$leg.yaml" 2>&1 | tee "$OUT/cfg_$leg.txt" \
    || die "腿 $leg config 產生失敗"
  ( cd "$SAM2REPO" && PYTHONPATH="$SAM2REPO" $PYT training/train.py \
      -c "configs/sam2.1_training/hsot_e31_leg_$leg.yaml" --use-cluster 0 --num-gpus 1 \
    ) > "$LOGS/leg_${leg}_console.log" 2>&1
  local rc=$?
  mark "LEG_${leg}_DONE"
  # D016：每腿一完成立即回傳，不等全部跑完（08-07 E05 全損的教訓）
  rclone copy "$LOGS" "$DEST/logs" --include "*.log" --transfers 8
  if [ $rc -ne 0 ]; then
    echo "⚠️ 腿 $leg 訓練非零退出（rc=$rc），最後 30 行："
    tail -30 "$LOGS/leg_${leg}_console.log"
    # 不 die：另一腿仍有價值，且判準能處理單腿失敗
  fi
}
run_leg A 5.0e-6
run_leg B 1.0e-4

# --- [5] 判準評定（判準寫在 e31_verdict_v1.py 檔頭，此處僅執行）----------------
echo "=== [5] 判準評定 ==="
PH1_ARG=""
[ -s ~/phase1_stageB.log ] && PH1_ARG="--phase1-log $HOME/phase1_stageB.log"
# 用 trainenv 的 python（判準需要 numpy；系統 python3 不保證有）
$PYT ~/e31_verdict_v1.py --leg-a "$LOGS/leg_A_console.log" \
  --leg-b "$LOGS/leg_B_console.log" $PH1_ARG \
  --json "$OUT/e31_verdict.json" 2>&1 | tee "$OUT/E31_VERDICT.txt"
mark E31_DONE
cp "$STAMP" "$OUT/e31_timing.txt"
rclone copy "$OUT" "$DEST/out" --transfers 8 || echo "⚠️ 最終回傳失敗"
rclone copy "$LOGS" "$DEST/logs" --include "*.log" --transfers 8

echo "✅ E31 完成。判決："; cat "$OUT/E31_VERDICT.txt" | tail -5
echo "🔻 依 D018 自我 terminate（id=$INSTANCE_ID）"
selfkill
