#!/usr/bin/env bash
# ============================================================================
# setup_gate01_d068_v1.sh — D068 Gate 0 + Gate 1：訓練保險可訓性檢驗（一機一次跑完）
#
# 授權：Sean 08-12 裁決選項 A（~$4；08-17 前出答案；無論結果 0.715 不重開；Gate 2 需再核可）
# 協定：5_outputs/strategy_research_20260812/report_audit_training.md §5（D068 草案）
# 判準：全部寫死於 gate0_events_v1.py / gate1_verdict_v2.py 檔頭（advisor 08-13 五修正已入）
#
# 【流程】G0 推論（SAMURAI backend、mask-cache）→ G0-A 恆等占比 ∧ G0-B 事件數 ⇒
#         rect 樹（base + x4 speed-aug + 事件 clip）→ prune → 50/50 file_list（assert 等量）→
#         G0-C lr=0 雙腿 loss 地板比值 ⇒ Gate 1（4 epochs、投影 loss、no-rotation、lr=1e-4）
# 【任一閘不過】die：回傳已產出物 + DIED.txt + terminate；不進下一閘（fail-closed）
#
# 🚨 開機前擋門（D067(f)）：本機先跑
#     gate1_verdict_v2 --mode selftest --neg-log <Phase1/E31b log> --marker ok.txt
#     並上傳 $DEST/preflight/verdict_selftest_ok.txt；本腳本 [0b] 查無此檔即拒跑。
# 🚨 terminate 一律用 INSTANCE_ID（Sean 08-12 指示：只關自己這台）
# 【紀律】D016 每階段 rclone｜D018 用完 terminate｜D036 單 tar｜計時器盤點（08-07 教訓）
# 【已知偏差（判決書必引）】G0 推論走 SAMURAI KF ⇒ 恆等占比偏高（G0-A 反保守）、
#   事件數偏低（G0-B 保守）；gate0_events_v1.py 檔頭同註。
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

INSTANCE_ID="${INSTANCE_ID:?必須傳入 INSTANCE_ID（terminate 用它，不用 name）}"
GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DEST="$GDRIVE/5_outputs/gate01_d068_20260813"
SAM2REPO=~/sam2repo
SAM2_SHA=2b90b9f5ceec907a1c18123530e92e794ad901a4     # 與 Phase 1／E31 同 SHA（D034）
SAMURAI=~/samurai
PYT=~/trainenv/bin/python
PY1=~/t1env/bin/python
EPOCHS_G1="${EPOCHS_G1:-4}"
LR_G1="${LR_G1:-1.0e-4}"        # E31b 已證此 LR 在恆等監督下無效 ⇒ 本次若動了＝監督換對了
OUT=~/g01_out; LOGS=~/g01_logs
STAMP=~/g01_timing.txt; : > "$STAMP"
mkdir -p "$OUT" "$LOGS" ~/ckpt

selfkill() {
  key=$(cat ~/.lambda_key | tr -d '\n'); [ -z "$key" ] && return 0
  curl -s -u "$key:" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H 'Content-Type: application/json' -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}"
}
die() {
  echo "🚨 $*"
  if command -v rclone >/dev/null 2>&1; then
    rclone copy "$OUT" "$DEST/out" --transfers 8 2>/dev/null
    rclone copy "$LOGS" "$DEST/logs" --include "*.log" --transfers 8 2>/dev/null
    echo "DIED: $*" > /tmp/g01_died.txt
    rclone copyto /tmp/g01_died.txt "$DEST/DIED.txt" 2>/dev/null
  else
    echo "（rclone 不在——死因無法上傳，僅本地留檔 /tmp/g01_died.txt）"
    echo "DIED: $*" > /tmp/g01_died.txt
  fi
  selfkill; exit 1
}
mark() { echo "$1 $(date +%s)" >> "$STAMP"; echo "⏱  $1 @ $(date +%H:%M:%S)"; }
mark G01_START

# --- [0] 計時器盤點 + 保險自毀（4h）-------------------------------------------
echo "=== [0] 計時器盤點 ==="
pgrep -af "sleep [0-9]+" || echo "（無既有計時器）"
nohup bash -c "sleep 14400
  key=\$(cat ~/.lambda_key | tr -d '\n'); [ -z \"\$key\" ] && exit 0
  curl -s -u \"\$key:\" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H 'Content-Type: application/json' -d '{\"instance_ids\":[\"$INSTANCE_ID\"]}'" \
  >/dev/null 2>&1 &
echo "保險自毀已掛（4h，PID=$!）"

# --- [0a] rclone 安裝（🚨 D067(f)④：全新機器沒有 rclone；v1 首航即死於此，$0.85）---
echo "=== [0a] rclone ==="
if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | sudo bash >/dev/null 2>&1 || die "rclone 安裝失敗"
fi
rclone lsf "$GDRIVE/1_data/" --max-depth 1 >/dev/null 2>&1 || die "rclone 無法讀 gDrive（conf 缺或壞）"
echo "rclone OK: $(rclone version | head -1)"

# --- [0b] 開機前擋門：verdict selftest marker ---------------------------------
echo "=== [0b] preflight：verdict selftest marker ==="
rclone copyto "$DEST/preflight/verdict_selftest_ok.txt" /tmp/selftest_ok.txt 2>/dev/null
[ -s /tmp/selftest_ok.txt ] || die "缺 verdict selftest marker——本機未過 D067(f) 自測，拒跑"

# --- [1] 背景拉資料（單 tar，D036）---------------------------------------------
echo "=== [1] 拉資料與工具 ==="
( set -e
  rclone copyto "$GDRIVE/1_data/packed/train_nir_fc.tar" ~/train_fc.tar
  mkdir -p ~/train_fc && tar -xf ~/train_fc.tar -C ~/train_fc && echo TRAINFC_READY
) > "$LOGS/pull_fc.log" 2>&1 &
PULL_FC=$!
rclone copyto "$GDRIVE/5_outputs/ef_phase1_20260811/davis_nir_ann.tar" ~/davis_nir_ann.tar \
  || die "davis ann tar 拉取失敗"
mkdir -p ~/davis_nir && tar -xf ~/davis_nir_ann.tar -C ~/davis_nir || die "ann tar 解開失敗"
rclone copyto "$GDRIVE/4_models/pretrained/sam2.1_hiera_large.pt" ~/ckpt/sam2.1_hiera_large.pt \
  || die "ckpt 拉取失敗"
rclone copyto "$GDRIVE/1_data/raw/2026training.csv" ~/2026training.csv || die "GT csv 拉取失敗"
rclone copyto "$GDRIVE/5_outputs/peft_phase0_20260810/peft_nir_train.txt" ~/peft_nir_train.txt \
  || die "train 名單拉取失敗"
for f in teacher_rect_v1.py make_train_cfg_v3.py loss_box_proj_v1.py gate0_events_v1.py \
         gate1_verdict_v2.py freeze_spec_v1.py prune_unannotated_v1.py; do
  rclone copyto "$GDRIVE/3_src/peft/$f" ~/"$f" || die "$f 拉取失敗"
done
rclone copyto "$GDRIVE/3_src/track_t1.py" ~/track_t1.py || die "track_t1.py 拉取失敗"

# --- [2] 環境 ×2（完全隔離 venv，CLAUDE.md 鐵律）-------------------------------
echo "=== [2a] trainenv + 官方 sam2 @ SHA ==="
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -d "$SAM2REPO" ] || git clone -q https://github.com/facebookresearch/sam2.git "$SAM2REPO"
( cd "$SAM2REPO" && git checkout -q "$SAM2_SHA" ) || die "sam2 SHA checkout 失敗"
[ -d ~/trainenv ] || uv venv --python 3.12 ~/trainenv
if [ ! -f ~/trainenv/.deps_done ]; then
  VIRTUAL_ENV=~/trainenv uv pip install -q torch torchvision --torch-backend=auto || die "torch 失敗"
  ( cd "$SAM2REPO" && VIRTUAL_ENV=~/trainenv uv pip install -q -e ".[dev]" ) || die "sam2[dev] 失敗"
  VIRTUAL_ENV=~/trainenv uv pip install -q submitit tensorboard opencv-python-headless pandas \
    || die "附屬失敗"
  touch ~/trainenv/.deps_done
fi
cp ~/freeze_spec_v1.py "$SAM2REPO/freeze_spec_v1.py"
cp ~/loss_box_proj_v1.py "$SAM2REPO/training/loss_box_proj_v1.py"
$PYT - <<'PY' || die "trainenv 驗證失敗"
import sys, torch; sys.path.insert(0, __import__("os").path.expanduser("~/sam2repo"))
import training.train, training.loss_box_proj_v1, freeze_spec_v1  # noqa
print(f"trainenv OK: torch {torch.__version__}  gpu={torch.cuda.get_device_name(0)}")
PY

echo "=== [2b] t1env + SAMURAI fork（Gate 0 推論用）==="
[ -d "$SAMURAI" ] || git clone -q --depth 1 https://github.com/yangchris11/samurai.git "$SAMURAI"
[ -d ~/t1env ] || uv venv --python 3.12 ~/t1env
if [ ! -f ~/t1env/.deps_done ]; then
  VIRTUAL_ENV=~/t1env uv pip install -q torch torchvision --torch-backend=auto || die "t1 torch 失敗"
  # 完整依賴照抄 drill v2（v1 首航二號機死因＝漏 loguru；scipy/loguru 皆 SAMURAI 未宣告依賴）
  VIRTUAL_ENV=~/t1env uv pip install -q -e "$SAMURAI/sam2" \
    scipy loguru tqdm pandas pillow opencv-python-headless || die "samurai+附屬失敗"
  touch ~/t1env/.deps_done
fi
$PY1 -c "
import torch, numpy, pandas, scipy, loguru, sam2
from sam2.build_sam import build_sam2_video_predictor  # 驗 import 鏈，不 build
assert torch.cuda.is_available(), 'CUDA 不可用'
print(f't1env OK: torch {torch.__version__} | numpy {numpy.__version__}')
" || die "t1env 驗證失敗"

wait $PULL_FC; grep -q TRAINFC_READY "$LOGS/pull_fc.log" || die "train_fc tar 未就緒"
FIRST_SEQ=$(head -1 ~/peft_nir_train.txt)
CAND=$(find ~/train_fc -maxdepth 4 -type d -name "$FIRST_SEQ" -print -quit)
[ -n "$CAND" ] || die "train tar 內找不到 $FIRST_SEQ"
TRAIN_ROOT=$(dirname "$CAND")
echo "TRAIN_ROOT=$TRAIN_ROOT"
mark ENV_DONE

# --- [3] Gate 0 推論（~40 分）--------------------------------------------------
echo "=== [3] Gate 0：SAMURAI+SAM2.1-L 訓練池 76 支（mask-cache 開）==="
( cd "$SAMURAI/sam2" && $PY1 ~/track_t1.py --frames-root "$TRAIN_ROOT" \
    --seq-list ~/peft_nir_train.txt --gt-csv ~/2026training.csv \
    --out-dir "$OUT/g0" --backend samurai --samurai-dir "$SAMURAI" \
    --ckpt ~/ckpt/sam2.1_hiera_large.pt --mask-cache \
  ) > "$LOGS/g0_infer.log" 2>&1 || die "Gate 0 推論失敗（log 尾 30 行：$(tail -30 $LOGS/g0_infer.log)）"
PRED_CSV=$(find "$OUT/g0" -maxdepth 1 -name "*.csv" -print -quit)
[ -n "$PRED_CSV" ] || die "Gate 0 找不到輸出 csv"
mark G0_INFER_DONE
rclone copy "$OUT/g0" "$DEST/out/g0" --include "*.csv" --transfers 8

# --- [4] Gate 0 分析與裁決 ------------------------------------------------------
echo "=== [4] G0-A 恆等占比 ∧ G0-B 事件數 ==="
$PYT ~/gate0_events_v1.py --pred-csv "$PRED_CSV" --gt-csv ~/2026training.csv \
  --seq-list ~/peft_nir_train.txt --masks-dir "$OUT/g0/masks" \
  --ann-root ~/davis_nir/Annotations \
  --out-events "$OUT/events.json" --out-verdict "$OUT/gate0_verdict.json" \
  2>&1 | tee "$LOGS/gate0_analysis.log" || die "Gate 0 分析失敗"
rclone copy "$OUT" "$DEST/out" --include "*.json" --transfers 8
G0OK=$($PYT -c "
import json; v=json.load(open('$OUT/gate0_verdict.json'))
print('yes' if (v['G0A_pass'] is True and v['G0B_pass']) else 'no')")
[ "$G0OK" = "yes" ] || die "Gate 0 未過（見 gate0_verdict.json）——依事前判準結案，不進 Gate 1"
mark G0_VERDICT_PASS

# --- [5] rect 樹 + prune + file_lists -----------------------------------------
echo "=== [5] rect-target 樹（base + x4 + 事件 clip）==="
$PYT ~/teacher_rect_v1.py --gt-csv ~/2026training.csv --frames-root "$TRAIN_ROOT" \
  --seq-list ~/peft_nir_train.txt --out-dir ~/rect --speed-strides 4 \
  --events-json "$OUT/events.json" --manifest "$OUT/rect_manifest.json" \
  2>&1 | tee "$LOGS/rect.log" || die "rect 樹生成失敗"
$PYT - <<'PY' || die "變體清單生成失敗"
import json, pathlib
m = json.load(open(pathlib.Path.home()/"g01_out/rect_manifest.json"))
names = [v["name"] for v in m["variants"]]
(pathlib.Path.home()/"rect/all_variants.txt").write_text("\n".join(names) + "\n")
print(f"變體 {len(names)}")
PY
$PYT ~/prune_unannotated_v1.py --davis-root ~/rect --seq-list ~/rect/all_variants.txt \
  --min-frames 8 --out-list ~/rect/pruned.txt --stats-json "$OUT/rect_prune_stats.json" \
  || die "prune 失敗"
$PYT - <<'PY' || die "file_list 生成失敗"
# 50/50 混採：sampler 均勻抽 file_list 條目 ⇒ 條目數就是混合比控制桿（advisor 修正 #5b）
import itertools, pathlib
home = pathlib.Path.home()
kept = [s for s in (home/"rect/pruned.txt").read_text().split() if s]
evt = sorted(s for s in kept if "__e" in s and "__x" not in s)
bg  = sorted(s for s in kept if "__e" not in s)
assert evt, "事件 clip 全滅於 prune——不應發生"
ctrl = list(itertools.islice(itertools.cycle(bg), len(evt)))   # 確定性配平（無隨機）
assert len(evt) == len(ctrl)
(home/"rect/list_5050.txt").write_text("\n".join(evt + ctrl) + "\n")
(home/"rect/list_events.txt").write_text("\n".join(evt) + "\n")
(home/"rect/list_control.txt").write_text("\n".join(ctrl) + "\n")
print(f"events={len(evt)}  control={len(ctrl)}  5050={len(evt)+len(ctrl)}")
PY
mark RECT_DONE
rclone copy "$OUT" "$DEST/out" --include "*.json" --transfers 8

# --- [6] G0-C：lr=0 雙腿 loss 地板比值 ------------------------------------------
CFG_DIR="$SAM2REPO/sam2/configs/sam2.1_training"
run_train() {  # $1=名稱 $2=file_list $3=lr $4=epochs
  $PYT ~/make_train_cfg_v3.py --sam2-repo "$SAM2REPO" \
    --img-folder ~/rect/JPEGImages --gt-folder ~/rect/Annotations \
    --file-list "$2" --ckpt ~/ckpt/sam2.1_hiera_large.pt \
    --num-epochs "$4" --log-dir "$LOGS/$1" --num-workers 8 --save-freq 0 \
    --base-lr "$3" --align-protocol --loss-box-proj --no-rotation \
    --out "$CFG_DIR/hsot_g01_$1.yaml" > "$OUT/cfg_$1.txt" 2>&1 \
    || die "$1 config 失敗（$(tail -5 $OUT/cfg_$1.txt)）"
  ( cd "$SAM2REPO" && PYTHONPATH="$SAM2REPO" $PYT training/train.py \
      -c "configs/sam2.1_training/hsot_g01_$1.yaml" --use-cluster 0 --num-gpus 1 \
    ) > "$LOGS/${1}_console.log" 2>&1
  local rc=$?
  rclone copy "$LOGS" "$DEST/logs" --include "*.log" --transfers 8
  return $rc
}
echo "=== [6] G0-C：lr=0 事件腿 vs 對照腿（各 1 epoch）==="
run_train floorevt ~/rect/list_events.txt 0.0 1 || die "floor 事件腿訓練失敗"
mark FLOOR_EVT_DONE
run_train floorctl ~/rect/list_control.txt 0.0 1 || die "floor 對照腿訓練失敗"
mark FLOOR_CTL_DONE
$PYT ~/gate1_verdict_v2.py --mode floor --events-log "$LOGS/floorevt_console.log" \
  --control-log "$LOGS/floorctl_console.log" --json "$OUT/g0c_floor.json" \
  2>&1 | tee "$LOGS/g0c.log" || die "G0-C 評定失敗"
rclone copy "$OUT" "$DEST/out" --include "*.json" --transfers 8
G0C=$($PYT -c "import json;print(json.load(open('$OUT/g0c_floor.json'))['verdict'])")
[ "$G0C" = "PASS" ] || die "G0-C 未過（事件窗無足量可優化信號）——依事前判準結案"
mark G0C_PASS

# --- [7] Gate 1：4 epochs 投影 loss 訓練 ----------------------------------------
echo "=== [7] Gate 1：50/50 混採、lr=$LR_G1、$EPOCHS_G1 epochs ==="
run_train gate1 ~/rect/list_5050.txt "$LR_G1" "$EPOCHS_G1"
RC=$?
[ $RC -ne 0 ] && echo "⚠️ Gate 1 非零退出（rc=$RC），log 尾 30 行：" && tail -30 "$LOGS/gate1_console.log"
mark GATE1_DONE
$PYT ~/gate1_verdict_v2.py --mode verdict --log "$LOGS/gate1_console.log" \
  --json "$OUT/gate1_verdict.json" 2>&1 | tee "$OUT/GATE1_VERDICT.txt" \
  || echo "⚠️ 判準評定異常（log 可能不完整）——原始 log 已回傳，本機可重評"

# --- [8] 收尾 -------------------------------------------------------------------
cp "$STAMP" "$OUT/g01_timing.txt"
rclone copy "$OUT" "$DEST/out" --transfers 8 || echo "⚠️ 最終回傳失敗"
rclone copy "$LOGS" "$DEST/logs" --include "*.log" --transfers 8
echo "✅ Gate 0/1 完成。判決："; tail -5 "$OUT/GATE1_VERDICT.txt" 2>/dev/null
echo "🔻 依 D018 自我 terminate（id=$INSTANCE_ID）"
selfkill
