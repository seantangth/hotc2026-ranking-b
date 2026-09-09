#!/usr/bin/env bash
# ============================================================================
# setup_ef_phase1_v1.sh — E-F Phase 1：SAM2.1-L 假色 video PEFT（NIR 76 支）
#
# 設計定稿：5_outputs/peft_phase0_20260810/DESIGN.md（判準 G0–G3 事前寫死）
# 前置：同機已跑完 setup_coldstart_drill_v2.sh ⇒ 已有 t1env / ~/samurai /
#       ~/ckpt/sam2.1_hiera_large.pt / ~/newdata_fc（val65 假色）/ ~/hsot / rclone
#
# 【G0-4 已於 08-11 本機判定，不再是未知】upstream(2b90b9f) PalettisedPNGSegmentLoader
#   對無 PNG 幀是 KeyError 崩潰 ⇒ DESIGN §3 fallback 生效：DAVIS 樹只保留有標註影格
#   （prune_unannotated_v1.py），機上 G0 只做「一步訓練跑通」的確認性 smoke。
#
# 【診斷加碼（Sean 08-11 裁示）】diag19 = val_v1 NIR 19 支病灶機制讀數（基線=E02 val65）
# 【紀律】D016 每階段 rclone｜D018 用完 terminate｜計時器全數盤點後才動（08-07 教訓）
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DEST="$GDRIVE/5_outputs/ef_phase1_20260811"
CKPT_DEST="$GDRIVE/4_models/runs/ef_phase1_20260811"
PY1=~/t1env/bin/python        # SAMURAI + SAM2.1（推論/teacher）
PYT=~/trainenv/bin/python     # 官方 sam2 訓練 stack
SAMURAI=~/samurai
SAM2REPO=~/sam2repo
SAM2_SHA=2b90b9f5ceec907a1c18123530e92e794ad901a4
PEFT=~/peft; OUT=~/ef_out
TRAIN_MAX_MIN="${TRAIN_MAX_MIN:-75}"
INSTANCE_NAME="${INSTANCE_NAME:-hsot-drill2}"
STAMP=~/ef_timing.txt; : > "$STAMP"
mkdir -p "$PEFT" "$OUT" ~/ef_logs
die() {
  echo "🚨 $*"
  rclone copy "$OUT" "$DEST/out" --transfers 8 2>/dev/null
  rclone copy ~/ef_logs "$DEST/logs" --include "*console.log" --transfers 8 2>/dev/null
  MID=$(find ~/ef_logs -name "checkpoint.pt" -print -quit 2>/dev/null)
  [ -n "$MID" ] && rclone copyto "$MID" "$CKPT_DEST/mid_checkpoint_on_die.pt" 2>/dev/null
  exit 1
}
mark() { echo "$1 $(date +%s)" >> "$STAMP"; echo "⏱  $1 @ $(date +%H:%M:%S)"; }
mark EF_START

# --- [0] 計時器交接（08-07 教訓：所有計時器先盤點、兩套都要處理）--------------
echo "=== [0] 計時器盤點與交接 ==="
pgrep -af "sleep [0-9]+" || echo "（無既有計時器）"
pkill -f "[s]leep 14400" 2>/dev/null && echo "已解除 drill 的 4h 保險" || echo "drill 保險不在（可能已過/未掛）"
nohup bash -c "
  sleep 16200
  key=\$(cat ~/.lambda_key | tr -d '\n'); [ -z \"\$key\" ] && exit 0
  id=\$(curl -s -u \"\$key:\" https://cloud.lambda.ai/api/v1/instances | python3 -c \"
import json,sys
d=json.load(sys.stdin).get('data',[])
m=[i['id'] for i in d if i.get('name')=='$INSTANCE_NAME']
print(m[0] if m else '')\")
  [ -n \"\$id\" ] && curl -s -u \"\$key:\" -X POST \
    https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H 'Content-Type: application/json' -d \"{\\\"instance_ids\\\":[\\\"\$id\\\"]}\"
" > ~/ef_insurance.log 2>&1 &
echo "⏰ Phase1 保險自毀 4.5h 已掛（name=$INSTANCE_NAME）"

# --- [1] 背景拉資料 -----------------------------------------------------------
echo "=== [1] 背景拉資料（單 tar，D036）==="
( set -e; rclone copyto "$GDRIVE/1_data/packed/train_nir_fc.tar" ~/train_fc.tar
  mkdir -p ~/train_fc && tar -xf ~/train_fc.tar -C ~/train_fc && echo TRAINFC_READY
) > ~/trainfc.log 2>&1 &
TRAINFC_PID=$!
( set -e; rclone copyto "$GDRIVE/1_data/packed/t1test_fc_75.tar" ~/test_fc.tar
  mkdir -p ~/test_fc && tar -xf ~/test_fc.tar -C ~/test_fc && echo TESTFC_READY
) > ~/testfc.log 2>&1 &
TESTFC_PID=$!

rclone copyto "$GDRIVE/1_data/raw/2026training.csv" ~/2026training.csv || die "GT 拉取失敗"
rclone copy "$GDRIVE/5_outputs/peft_phase0_20260810/" "$PEFT/lists/" --include "*.txt" || die "清單拉取失敗"
for f in peft_nir_train.txt peft_nir_holdout.txt peft_nir_eval_local.txt peft_nir_eval_diag_v1.txt; do
  [ -s "$PEFT/lists/$f" ] || die "清單缺 $f"
done
rclone copyto "$GDRIVE/3_src/peft/teacher_davis_v2.py"      ~/teacher_davis_v2.py
rclone copyto "$GDRIVE/3_src/peft/freeze_spec_v1.py"        ~/freeze_spec_v1.py
rclone copyto "$GDRIVE/3_src/peft/make_train_cfg_v1.py"     ~/make_train_cfg_v1.py
rclone copyto "$GDRIVE/3_src/peft/prune_unannotated_v1.py"  ~/prune_unannotated_v1.py
rclone copyto "$GDRIVE/3_src/peft/g1_report_v1.py"          ~/g1_report_v1.py
rclone copyto "$GDRIVE/5_outputs/submissions/exp003_samurai_large.csv" ~/e02_test.csv
rclone copyto "$GDRIVE/5_outputs/submissions/exp004_dam4sam_large.csv" ~/e03_test.csv
rclone copyto "$GDRIVE/5_outputs/t1_rerun_20260805/submission.csv" ~/e02_val65.csv
for f in ~/teacher_davis_v2.py ~/freeze_spec_v1.py ~/make_train_cfg_v1.py \
         ~/prune_unannotated_v1.py ~/g1_report_v1.py ~/e02_test.csv ~/e03_test.csv ~/e02_val65.csv; do
  [ -s "$f" ] || die "缺 $f"
done
# v2-NIR 7 = eval_local − holdout（幀在 val65 tar，已由 drill 解開）
python3 - <<'PY'
from pathlib import Path
el = [s for s in Path.home().joinpath("peft/lists/peft_nir_eval_local.txt").read_text().split() if s]
hold = set(Path.home().joinpath("peft/lists/peft_nir_holdout.txt").read_text().split())
v27 = [s for s in el if s not in hold]
assert len(v27) == 7, f"v2-NIR 應為 7 支, 得 {len(v27)}"
Path.home().joinpath("peft/lists/peft_nir_v2nir7.txt").write_text("\n".join(v27) + "\n")
print("v2nir7:", v27)
PY
[ -d ~/newdata_fc ] || die "~/newdata_fc 不在（drill 未跑？）"
[ -f ~/ckpt/sam2.1_hiera_large.pt ] || die "SAM2.1-L 權重不在"

# --- [2] trainenv（第三個完全隔離 venv）+ 官方 sam2 repo（釘 SHA）------------
echo "=== [2] trainenv + 官方 sam2 訓練 stack ==="
mark ENV_START
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d "$SAM2REPO" ] || git clone -q https://github.com/facebookresearch/sam2.git "$SAM2REPO"
( cd "$SAM2REPO" && git checkout -q "$SAM2_SHA" ) || die "sam2 SHA checkout 失敗"
echo "sam2repo SHA = $SAM2_SHA" > "$OUT/pins.txt"
[ -d ~/trainenv ] || uv venv --python 3.12 ~/trainenv
if [ ! -f ~/trainenv/.deps_done ]; then
  VIRTUAL_ENV=~/trainenv uv pip install -q torch torchvision --torch-backend=auto || die "torch 安裝失敗"
  ( cd "$SAM2REPO" && VIRTUAL_ENV=~/trainenv uv pip install -q -e ".[dev]" ) || die "sam2[dev] 安裝失敗"
  VIRTUAL_ENV=~/trainenv uv pip install -q submitit tensorboard opencv-python-headless || die "訓練附屬依賴失敗"
  touch ~/trainenv/.deps_done || die "deps_done 標記失敗（S-05）"
fi
cp ~/freeze_spec_v1.py "$SAM2REPO/freeze_spec_v1.py"
$PYT - <<'PY' || die "trainenv 驗證失敗"
import torch, sam2, hydra, omegaconf, fvcore, submitit
assert torch.cuda.is_available()
import sys; sys.path.insert(0, __import__("os").path.expanduser("~/sam2repo"))
import training.train  # noqa
import freeze_spec_v1  # noqa
print(f"trainenv OK: torch {torch.__version__}")
PY
mark ENV_DONE

wait $TRAINFC_PID; grep -q TRAINFC_READY ~/trainfc.log || die "train_nir_fc 拉取失敗"
# tar 內層結構自動偵測（頂層即序列目錄，或多包一層）
TRAIN_ROOT=~/train_fc
FIRST_SEQ=$(head -1 "$PEFT/lists/peft_nir_train.txt")
if [ ! -d "$TRAIN_ROOT/$FIRST_SEQ" ]; then
  CAND=$(find ~/train_fc -maxdepth 2 -type d -name "$FIRST_SEQ" -print -quit)
  [ -n "$CAND" ] || die "train tar 內找不到 $FIRST_SEQ"
  TRAIN_ROOT=$(dirname "$CAND")
fi
echo "TRAIN_ROOT=$TRAIN_ROOT"
# 防 D059 fallback 靜默填充：清單內每支序列的影格目錄必須真的存在
MISS=0
while read -r s; do [ -n "$s" ] && [ ! -d "$TRAIN_ROOT/$s" ] && { echo "缺 $TRAIN_ROOT/$s"; MISS=1; }; done \
  < <(cat "$PEFT/lists/peft_nir_train.txt" "$PEFT/lists/peft_nir_holdout.txt")
while read -r s; do [ -n "$s" ] && [ ! -d "$HOME/newdata_fc/$s" ] && { echo "缺 newdata_fc/$s"; MISS=1; }; done \
  < <(cat "$PEFT/lists/peft_nir_v2nir7.txt" "$PEFT/lists/peft_nir_eval_diag_v1.txt")
[ "$MISS" -eq 0 ] || die "序列目錄缺失（見上），不得讓 exception fallback 靜默填充"

# --- [3] teacher mask 重生 v2（t1env）→ DAVIS 樹 → fallback 修剪 ---------------
echo "=== [3] teacher regen v2（stride1、floor 0.5）==="
mark TEACHER_START
$PY1 ~/teacher_davis_v2.py --frames-root "$TRAIN_ROOT" --gt-csv ~/2026training.csv \
  --seq-list "$PEFT/lists/peft_nir_train.txt" --samurai-dir "$SAMURAI" \
  --ckpt ~/ckpt/sam2.1_hiera_large.pt --out-dir ~/davis_nir --tightness-floor 0.5 \
  2>&1 | tail -20 || die "teacher regen 失敗"
$PY1 ~/prune_unannotated_v1.py --davis-root ~/davis_nir \
  --seq-list "$PEFT/lists/peft_nir_train.txt" --min-frames 8 \
  --out-list ~/davis_nir/train_list_pruned.txt \
  --stats-json "$OUT/teacher_v2_stats.json" || die "prune 失敗"
NSEQ_TRAIN=$(grep -c . ~/davis_nir/train_list_pruned.txt)
[ "$NSEQ_TRAIN" -ge 40 ] || die "修剪後只剩 $NSEQ_TRAIN 支（<40，資料層異常）"
mark TEACHER_DONE
rclone copy "$OUT" "$DEST/out" --transfers 8
( cd ~/davis_nir && tar -chf ~/davis_nir_ann.tar Annotations meta train_list_pruned.txt \
  && rclone copyto ~/davis_nir_ann.tar "$DEST/davis_nir_ann.tar" ) &
ANN_UP_PID=$!

# --- [4] G0 smoke：stageA（1 epoch）+ 四項檢查 --------------------------------
echo "=== [4] G0 smoke（stageA 1 epoch）==="
mark G0_START
CFG_DIR="$SAM2REPO/sam2/configs/sam2.1_training"
$PYT ~/make_train_cfg_v1.py --sam2-repo "$SAM2REPO" \
  --img-folder ~/davis_nir/JPEGImages --gt-folder ~/davis_nir/Annotations \
  --file-list ~/davis_nir/train_list_pruned.txt --ckpt ~/ckpt/sam2.1_hiera_large.pt \
  --num-epochs 1 --log-dir ~/ef_logs/stageA --num-workers 8 \
  --out "$CFG_DIR/hsot_ef_stageA.yaml" || die "stageA config 失敗"

# G0-4 確認性 probe：pruned 樹上 loader 取樣一筆、逐幀有 mask（本機已判定，機上復核）
( cd "$SAM2REPO" && PYTHONPATH="$SAM2REPO" $PYT - <<'PY' ) || die "G0-4 loader probe 失敗"
import os
from training.dataset.vos_raw_dataset import PNGRawDataset
H = os.path.expanduser
ds = PNGRawDataset(img_folder=H("~/davis_nir/JPEGImages"), gt_folder=H("~/davis_nir/Annotations"),
                   file_list_txt=H("~/davis_nir/train_list_pruned.txt"))
video, loader = ds.get_video(0)
fids = [f.frame_idx for f in video.frames][:12]
segs = [len(loader.load(fid)) for fid in fids]
assert all(s >= 1 for s in segs), f"有幀無 mask：{list(zip(fids, segs))}"
print(f"G0-4 ✅ {video.video_name}: 前12幀 fid={fids} 皆有 mask（fallback 樹成立）")
PY

mark STAGEA_TRAIN_START
( cd "$SAM2REPO" && PYTHONPATH="$SAM2REPO" $PYT training/train.py \
    -c configs/sam2.1_training/hsot_ef_stageA.yaml --use-cluster 0 --num-gpus 1 \
  ) > ~/ef_logs/stageA_console.log 2>&1 || { tail -40 ~/ef_logs/stageA_console.log; die "stageA 訓練失敗"; }
mark STAGEA_TRAIN_DONE
STAGEA_CKPT=$(find ~/ef_logs/stageA -name "checkpoint.pt" -print -quit)
[ -n "$STAGEA_CKPT" ] || die "stageA 無 checkpoint.pt"
grep -q "freeze_spec_v1" ~/ef_logs/stageA_console.log || die "凍結報告未出現（freeze 沒接上）"
grep "可訓練" ~/ef_logs/stageA_console.log | head -2
PCT=$($PYT - <<'PY'
import re
log = open(__import__("os").path.expanduser("~/ef_logs/stageA_console.log")).read()
m = re.search(r"可訓練 ([\d.]+)M \(([\d.]+)%\)", log)
print(m.group(2) if m else "NA")
PY
)
[ "$PCT" != "NA" ] || die "凍結比例解析失敗"
python3 -c "p=float('$PCT'); assert 3.0 <= p <= 12.0, f'可訓練比例 {p}% 超出 3-12% 預期'" \
  || die "凍結比例異常（$PCT%）"

# G0-2.5：state_dict 級斷言（比「輸出不同」強——直接驗免 merge 宣稱與凍結不變量）
PYTHONPATH="$SAM2REPO" $PYT - <<'PY' || die "G0-2.5 state_dict 斷言失敗"
import os, torch
import freeze_spec_v1 as fz
H = os.path.expanduser
import glob
stagea = glob.glob(H("~/ef_logs/stageA/**/checkpoint.pt"), recursive=True)[0]
a = torch.load(H("~/ckpt/sam2.1_hiera_large.pt"), map_location="cpu", weights_only=True)["model"]
b = torch.load(stagea, map_location="cpu", weights_only=False)["model"]
ka, kb = set(a), set(b)
assert ka == kb, f"鍵集不同：缺 {sorted(ka-kb)[:3]} 多 {sorted(kb-ka)[:3]}（免 merge 宣稱破功）"
changed = [k for k in ka if not torch.equal(a[k], b[k])]
bad = [k for k in changed if not fz.is_trainable(k)]
enc_changed = [k for k in changed if k.startswith("image_encoder.")]
assert not enc_changed, f"image_encoder 有 {len(enc_changed)} 鍵被動到：{enc_changed[:3]}"
assert not bad, f"{len(bad)} 個變動鍵落在凍結區之外：{bad[:5]}"
assert changed, "沒有任何鍵變動（訓練沒生效）"
print(f"G0-2.5 ✅ 鍵集相同；變動 {len(changed)} 鍵全部落在 freeze_spec 可訓練前綴內；image_encoder 位元不動")
PY

# G0-2/3：stageA ckpt 載入 SAMURAI fork、單支輸出與 base 不同（防 E18 型沒接上）
G0SEQ=$(head -1 "$PEFT/lists/peft_nir_holdout.txt")
echo "$G0SEQ" > ~/g0_seq.txt
$PY1 ~/track_t1.py --frames-root "$TRAIN_ROOT" --seq-list ~/g0_seq.txt --out-dir ~/g0_base \
  --backend samurai --samurai-dir "$SAMURAI" --ckpt ~/ckpt/sam2.1_hiera_large.pt \
  --gt-csv ~/2026training.csv || die "G0-2 base 腿失敗"
$PY1 ~/track_t1.py --frames-root "$TRAIN_ROOT" --seq-list ~/g0_seq.txt --out-dir ~/g0_peft \
  --backend samurai --samurai-dir "$SAMURAI" --ckpt "$STAGEA_CKPT" \
  --gt-csv ~/2026training.csv || die "G0-2 stageA ckpt 載入 SAMURAI fork 失敗"
python3 - <<'PY' || die "G0-3 失敗：stageA 輸出與 base 完全相同（權重沒生效）"
import pandas as pd, os
H = os.path.expanduser
a = pd.read_csv(H("~/g0_base/submission.csv")).set_index("ID")
b = pd.read_csv(H("~/g0_peft/submission.csv")).set_index("ID")
assert len(a) == len(b) and (a.index == b.index).all(), "兩腿 ID 集不一致"
diff = (a[["x","y","width","height"]].round(2) != b[["x","y","width","height"]].round(2)).any(axis=1).sum()
print(f"G0-3 ✅ {diff}/{len(a)} 幀輸出不同（權重已生效）")
assert diff > 0
PY
mark G0_DONE
rclone copy ~/ef_logs "$DEST/logs" --include "*.log" --transfers 8

# --- [5] 主訓練 stageB（epochs 由 stageA 實測 step time 回填，上限 40）--------
echo "=== [5] stageB 主訓練 ==="
SEC_A=$(python3 - <<PY
t = {k: int(v) for k, v in (l.split() for l in open("$STAMP").read().splitlines() if l.strip())}
print(t["STAGEA_TRAIN_DONE"] - t["STAGEA_TRAIN_START"])
PY
)
STEPS=$((2 * NSEQ_TRAIN))
EPOCHS=$(python3 -c "
sec_epoch = max($SEC_A - 90, 60)   # 扣建構/載重開銷（保守）
n = int(($TRAIN_MAX_MIN * 60) // sec_epoch)
print(max(1, min(n, 40)))")
echo "stageA 1 epoch 實測 ${SEC_A}s（$STEPS steps）⇒ stageB epochs=$EPOCHS（上限 ${TRAIN_MAX_MIN} 分）"
echo "stageA_epoch_sec=$SEC_A stageB_epochs=$EPOCHS" >> "$OUT/pins.txt"
$PYT ~/make_train_cfg_v1.py --sam2-repo "$SAM2REPO" \
  --img-folder ~/davis_nir/JPEGImages --gt-folder ~/davis_nir/Annotations \
  --file-list ~/davis_nir/train_list_pruned.txt --ckpt ~/ckpt/sam2.1_hiera_large.pt \
  --num-epochs "$EPOCHS" --log-dir ~/ef_logs/stageB --num-workers 8 --save-freq 1 \
  --out "$CFG_DIR/hsot_ef_stageB.yaml" || die "stageB config 失敗"
mark TRAIN_START
# D016：中途 ckpt 每 10 分鐘滾動回傳（最後完成的 epoch 永遠已落地，硬殺也不全損）
( while sleep 600; do
    MID=$(find ~/ef_logs/stageB -name "checkpoint.pt" -print -quit 2>/dev/null)
    [ -n "$MID" ] && rclone copyto "$MID" "$CKPT_DEST/mid_checkpoint.pt" 2>/dev/null
  done ) & MIDUP_PID=$!
( cd "$SAM2REPO" && PYTHONPATH="$SAM2REPO" $PYT training/train.py \
    -c configs/sam2.1_training/hsot_ef_stageB.yaml --use-cluster 0 --num-gpus 1 \
  ) > ~/ef_logs/stageB_console.log 2>&1 || { kill $MIDUP_PID 2>/dev/null; tail -40 ~/ef_logs/stageB_console.log; die "stageB 訓練失敗"; }
kill $MIDUP_PID 2>/dev/null
mark TRAIN_DONE
EF_CKPT=$(find ~/ef_logs/stageB -name "checkpoint.pt" -print -quit)
[ -n "$EF_CKPT" ] || die "stageB 無 checkpoint.pt"
cp "$EF_CKPT" ~/ckpt/ef_phase1.pt
rclone copyto ~/ckpt/ef_phase1.pt "$CKPT_DEST/ef_phase1.pt" || die "ckpt 回傳失敗（D022）"
rclone copy ~/ef_logs "$DEST/logs" --include "*.log" --transfers 8
echo "✅ stageB 完成，ckpt 已回傳 gDrive"

# --- [6] G1 兩腿（holdout26 + v2nir7）＋ diag19 -------------------------------
echo "=== [6] G1 兩腿推論 + diag19 ==="
mark G1_START
run_leg() { # $1=ckpt $2=root $3=list $4=outdir
  $PY1 ~/track_t1.py --frames-root "$2" --seq-list "$3" --out-dir "$4" \
    --backend samurai --samurai-dir "$SAMURAI" --ckpt "$1" \
    --gt-csv ~/2026training.csv || die "leg 失敗: $4"
}
run_leg ~/ckpt/sam2.1_hiera_large.pt "$TRAIN_ROOT" "$PEFT/lists/peft_nir_holdout.txt" ~/leg_base_hold
run_leg ~/ckpt/ef_phase1.pt         "$TRAIN_ROOT" "$PEFT/lists/peft_nir_holdout.txt" ~/leg_peft_hold
run_leg ~/ckpt/sam2.1_hiera_large.pt ~/newdata_fc "$PEFT/lists/peft_nir_v2nir7.txt"  ~/leg_base_v27
run_leg ~/ckpt/ef_phase1.pt          ~/newdata_fc "$PEFT/lists/peft_nir_v2nir7.txt"  ~/leg_peft_v27
python3 - <<'PY'
import pandas as pd, os
H = os.path.expanduser
for tag in ["base", "peft"]:
    a = pd.read_csv(H(f"~/leg_{tag}_hold/submission.csv"))
    b = pd.read_csv(H(f"~/leg_{tag}_v27/submission.csv"))
    pd.concat([a, b]).to_csv(H(f"~/g1_{tag}33.csv"), index=False)
print("g1 concat OK")
PY
run_leg ~/ckpt/ef_phase1.pt ~/newdata_fc "$PEFT/lists/peft_nir_eval_diag_v1.txt" ~/leg_peft_diag19
mark G1_LEGS_DONE
rclone copy ~/g1_base33.csv "$DEST/out" && rclone copy ~/g1_peft33.csv "$DEST/out"
rclone copyto ~/leg_peft_diag19/submission.csv "$DEST/out/diag19_peft.csv"

set +e
$PY1 ~/g1_report_v1.py --gt-csv ~/2026training.csv \
  --eval-seqs "$PEFT/lists/peft_nir_eval_local.txt" \
  --base-csv ~/g1_base33.csv --peft-csv ~/g1_peft33.csv \
  --diag-seqs "$PEFT/lists/peft_nir_eval_diag_v1.txt" \
  --diag-base-csv ~/e02_val65.csv --diag-peft-csv ~/leg_peft_diag19/submission.csv \
  --out-json "$OUT/g1_g2_diag_report.json" 2>&1 | tee "$OUT/g1_report.txt"
G1_RC=${PIPESTATUS[0]}
mark G1_DONE
rclone copy "$OUT" "$DEST/out" --transfers 8
[ "$G1_RC" -eq 0 ] || [ "$G1_RC" -eq 3 ] || die "g1_report 異常退出 rc=$G1_RC"

# --- [7] G3 test-NIR 腿（供明日零 GPU 拼接 LB；G1 災難則跳過）------------------
if [ "$G1_RC" -eq 3 ]; then
  echo "🚨 G1 災難旗標觸發 ⇒ 跳過 G3 test 腿（依 DESIGN §5 結案路徑）"
else
  echo "=== [7] G3 test-NIR 腿（v011 結構、唯一變因＝ckpt）==="
  mark G3_START
  wait $TESTFC_PID; grep -q TESTFC_READY ~/testfc.log || die "test_fc 拉取失敗"
  TEST_ROOT=~/test_fc
  if [ ! -d "$TEST_ROOT/nir-bee2" ]; then
    CAND=$(find ~/test_fc -maxdepth 2 -type d -name "nir-bee2" -print -quit)
    [ -n "$CAND" ] || die "test tar 內找不到 nir-bee2"
    TEST_ROOT=$(dirname "$CAND")
  fi
  find "$TEST_ROOT" -mindepth 1 -maxdepth 1 -type d -name "nir-*" -printf "%f\n" | sort > ~/nir22.txt
  N22=$(grep -c . ~/nir22.txt); echo "test NIR 序列 = $N22 支"
  [ "$N22" -ge 20 ] || die "test NIR 只有 $N22 支（異常）"
  # 窗＝E02∪E03 @0.40（v011 同構、零 SAM3 成分）
  PYTHONPATH=~ $PY1 -m hsot.crop_rerun prep --frames-root "$TEST_ROOT" \
    --base-csv ~/e02_test.csv --envelope-extra ~/e03_test.csv --area-frac-max 0.40 \
    --out-root ~/crop_fc_test --meta ~/crop_meta_test.json | tail -8 || die "G3 crop prep 失敗"
  python3 - <<'PY' || die "G3 meta NIR 過濾失敗"
import json, os
H = os.path.expanduser
m = json.load(open(H("~/crop_meta_test.json")))
nir = {k: v for k, v in m.items() if k.startswith("nir-")}
json.dump(nir, open(H("~/crop_meta_nir.json"), "w"))
open(H("~/nir_crop_seqs.txt"), "w").write("\n".join(sorted(nir)) + "\n")
print(f"crop 選中 {len(m)} 支，其中 NIR {len(nir)} 支: {sorted(nir)}")
PY
  $PY1 ~/track_t1.py --frames-root "$TEST_ROOT" --seq-list ~/nir22.txt --out-dir ~/g3_base_nir \
    --backend samurai --samurai-dir "$SAMURAI" --ckpt ~/ckpt/ef_phase1.pt \
    || die "G3 base 腿失敗"
  if [ -s ~/nir_crop_seqs.txt ] && [ "$(grep -c . ~/nir_crop_seqs.txt)" -ge 1 ]; then
    $PY1 ~/track_t1.py --frames-root ~/crop_fc_test --seq-list ~/nir_crop_seqs.txt \
      --out-dir ~/g3_crop_nir --backend samurai --samurai-dir "$SAMURAI" \
      --ckpt ~/ckpt/ef_phase1.pt || die "G3 crop 腿失敗"
    PYTHONPATH=~ $PY1 -m hsot.crop_rerun merge --base-csv ~/g3_base_nir/submission.csv \
      --crop-csv ~/g3_crop_nir/submission.csv --meta ~/crop_meta_nir.json \
      --out "$OUT/g3_nir_peft.csv" || die "G3 merge 失敗"
  else
    cp ~/g3_base_nir/submission.csv "$OUT/g3_nir_peft.csv"
  fi
  cp ~/crop_meta_nir.json "$OUT/" 2>/dev/null || true
  mark G3_DONE
  rclone copy "$OUT" "$DEST/out" --transfers 8
fi

# --- [8] 收尾 ------------------------------------------------------------------
wait $ANN_UP_PID 2>/dev/null || true
mark EF_DONE
echo "=== [8] 時間帳 ==="
python3 - <<PY
t = {k: int(v) for k, v in (l.split() for l in open("$STAMP").read().splitlines() if l.strip())}
def span(a, b, lab):
    if a in t and b in t:
        s = t[b] - t[a]; print(f"  {lab:34s} {s//60:3d} 分 {s%60:02d} 秒")
span("EF_START","ENV_DONE","環境+資料")
span("TEACHER_START","TEACHER_DONE","teacher regen v2 + prune")
span("G0_START","G0_DONE","G0 smoke（含 stageA 1 epoch）")
span("TRAIN_START","TRAIN_DONE","stageB 主訓練")
span("G1_START","G1_DONE","G1 兩腿 + diag19 + 報告")
span("G3_START","G3_DONE","G3 test-NIR 腿")
span("EF_START","EF_DONE","★ Phase 1 總計")
PY
rclone copy "$OUT" "$DEST/out" --transfers 8
rclone copyto "$STAMP" "$DEST/ef_timing.txt"
rclone copy ~/ef_logs "$DEST/logs" --include "*console.log" --transfers 8
echo "EF_PHASE1_DONE"
echo "⚠️ 4.5h 保險自毀仍掛著；外層 wrapper 會 rclone 全部 log 後 terminate（D018）"
