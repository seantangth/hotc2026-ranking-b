#!/usr/bin/env bash
# ============================================================================
# setup_coldstart_drill3_v1.sh — 冷啟動演練 #3：**v049 雙管線交付鏈端到端重現**
#
# 【與 #1／#2 的差異】
#   #1＝同一批資料能重跑出同樣結果（單套管線 v008）；#2＝換一批資料還能跑（泛化）。
#   #3 要驗的是 **08-16 才定案的新交付鏈**（D070/D071/D073）：
#     兩套基礎管線（e02 SAMURAI ＋ e15 SAM3）→ crop prep【一次】→ 兩 backend 各自重跑/merge
#     → finalize_submission.py（corr both → splice K=6 → validate）
#   這條鏈**從未被端到端跑過**——v023/v012 是 08-06 分兩次跑的，splice 與 finalize 是離線拼的。
#
# 【驗收（事前寫死）】對 15 支代表性子集，最終檔須與 `sub_v049_src_v012_K6.csv` 的
#   對應列**位元級相同**。任一列不符即 exit 3 並列出前 5 筆差異。
#   子集涵蓋：splice 大戶(nir-motorcycle4 218 幀)／極小目標(crop 必觸發)／
#   大目標對照(crop 不觸發)／三模態齊全。
#
# 【關鍵規格（勿改）】
#   ✗ **不傳 --gt-csv**——test 與 9/7 新資料的 ID 皆為每序列 1-based；
#     傳了會走 training 的全域幀號分支（track_t1.py:70-78），測到的是錯的那條路。
#   ✓ crop 窗只算【一次】：`prep --base-csv e15 --envelope-extra e02`（D073④）。
#     兩個 backend 共用該窗，只有 merge 的 base 不同（SAM3→e15／SAMURAI→e02）。
#
# 【昨日教訓已內建】①新機沒有 rclone，第一個 rclone 前必裝（D067(f)④）
#   ②環境依賴整段抄 drill v2，不抄一半（漏 loguru 害死過一台）
#   ③每階段完成即 rclone 回傳（D016：08-07 曾因最後才傳而蒸發整輪產出）
# 🚨 terminate 一律用傳入的 INSTANCE_ID（只關自己開的機器）
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

INSTANCE_ID="${INSTANCE_ID:?必須傳入 INSTANCE_ID}"
GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
SAM3_CKPT_SHA256=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e

# 官方 facebook/sam3 是 gated repo：權杖缺失就在開工前停，不要跑到背景下載才失敗。
: "${HF_TOKEN:?HF_TOKEN is required (gated facebook/sam3); export it before running}"
DEST="$GDRIVE/5_outputs/coldstart_drill3_20260816"
PY1=~/t1env/bin/python; PY3=~/sam3env/bin/python
SAMURAI=~/samurai
FC=~/test_fc; OUT=~/drill3; mkdir -p "$OUT"
STAMP="$OUT/timing.txt"; : > "$STAMP"
mark() { echo "$1 $(date +%s)" >> "$STAMP"; echo "⏱  $1 @ $(date +%H:%M:%S)"; }

selfkill() {
  key=$(cat ~/.lambda_key | tr -d '\n'); [ -z "$key" ] && return 0
  curl -s -u "$key:" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H 'Content-Type: application/json' -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}"
}
die() {
  echo "🚨 $*"
  if command -v rclone >/dev/null 2>&1; then
    echo "DIED: $*" > "$OUT/DIED.txt"; rclone copy "$OUT" "$DEST" --transfers 8 2>/dev/null
  fi
  selfkill; exit 1
}

mark DRILL3_START
# --- [0] 保險自毀（150 分；預估 100 分）--------------------------------------
pgrep -af "sleep [0-9]+" || echo "（無既有計時器）"
nohup bash -c "sleep 9000
  key=\$(cat ~/.lambda_key | tr -d '\n'); [ -z \"\$key\" ] && exit 0
  curl -s -u \"\$key:\" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H 'Content-Type: application/json' -d '{\"instance_ids\":[\"$INSTANCE_ID\"]}'" >/dev/null 2>&1 &
echo "保險自毀已掛（150 分）"

# --- [0a] rclone（新機沒有）---------------------------------------------------
if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | sudo bash >/dev/null 2>&1 || die "rclone 安裝失敗"
fi
rclone lsf "$GDRIVE/1_data/" --max-depth 1 >/dev/null 2>&1 || die "rclone 讀不到 gDrive"

# --- [1] 背景：test 假色（單 tar，D036）--------------------------------------
( set -e
  rclone copyto "$GDRIVE/1_data/packed/t1test_fc_75.tar" ~/test.tar
  mkdir -p "$FC" && tar -xf ~/test.tar -C "$FC"
  echo "FC_READY seqs=$(find "$FC" -mindepth 1 -maxdepth 1 -type d | wc -l)"
) > ~/fc.log 2>&1 &
FC_PID=$!

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
mkdir -p ~/ckpt

# --- [2a/2b] 背景：兩份權重 ---------------------------------------------------
( set -e
  : "${HF_TOKEN:?HF_TOKEN is required for the gated facebook/sam3 weight}"
  [ -f ~/ckpt/sam3.pt ] || curl -fL -H "Authorization: Bearer $HF_TOKEN" \
    -o ~/ckpt/sam3.pt "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt"
  echo "$SAM3_CKPT_SHA256  $HOME/ckpt/sam3.pt" | sha256sum -c -
  [ "$(stat -c%s ~/ckpt/sam3.pt)" -gt 3400000000 ] || { echo "🚨 SAM3 權重太小"; exit 1; }
  echo CKPT3_READY ) > ~/ckpt3.log 2>&1 &
C3=$!
( set -e
  [ -f ~/ckpt/sam2.1_hiera_large.pt ] || rclone copy "$GDRIVE/4_models/pretrained/sam2.1_hiera_large.pt" ~/ckpt/
  [ "$(stat -c%s ~/ckpt/sam2.1_hiera_large.pt)" -gt 800000000 ] || { echo "🚨 SAM2.1 權重太小"; exit 1; }
  echo CKPT2_READY ) > ~/ckpt2.log 2>&1 &
C2=$!

# --- [2c] t1env（完全隔離；scipy/loguru 是 SAMURAI 未宣告依賴）----------------
[ -d ~/t1env ] || uv venv --python 3.12 ~/t1env
[ -d "$SAMURAI" ] || git clone --depth 1 https://github.com/yangchris11/samurai.git "$SAMURAI"
if [ ! -f ~/t1env/.deps_done ]; then
  VIRTUAL_ENV=~/t1env uv pip install -q torch torchvision --torch-backend=auto || die "t1env torch 失敗"
  VIRTUAL_ENV=~/t1env uv pip install -q -e "$SAMURAI/sam2" \
    scipy loguru tqdm pandas pillow opencv-python-headless || die "t1env deps 失敗"
  touch ~/t1env/.deps_done
fi
$PY1 -c "
import torch, numpy, pandas, scipy, sam2
assert torch.cuda.is_available(), 'CUDA 不可用'
print(f't1env OK: torch {torch.__version__} | {torch.cuda.get_device_name(0)}')" || die "t1env 驗證失敗"

# --- [2d] sam3env（釘 SHA，D059⑤）--------------------------------------------
SAM3_SHA=96914d2425f90a64f45ca977c2b5165418099543
[ -d ~/sam3env ] || uv venv --python 3.12 ~/sam3env
if [ ! -f ~/sam3env/.deps_done ]; then
  VIRTUAL_ENV=~/sam3env uv pip install -q torch torchvision --torch-backend=auto || die "sam3env torch 失敗"
  VIRTUAL_ENV=~/sam3env uv pip install -q "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}" \
    "numpy<2" ftfy==6.1.1 regex einops psutil scipy av \
    pycocotools numba python-rapidjson pandas pillow tqdm opencv-python-headless || die "sam3env deps 失敗"
  VIRTUAL_ENV=~/sam3env uv pip install -q "setuptools<81" || die "sam3env setuptools 失敗"
  touch ~/sam3env/.deps_done
fi
$PY3 -c "
import sam3, numpy
assert numpy.__version__.startswith('1.'), f'numpy {numpy.__version__} 應 <2'
print('sam3env OK')" 2>&1 | grep -viE 'warning|deprecat|^ +import' || die "sam3env 驗證失敗"

# --- [2e] 交付物程式碼 ＋ 子集清單 ＋ 參考檔 -----------------------------------
rclone copyto "$GDRIVE/3_src/track_t1.py" ~/track_t1.py || die "track_t1.py 缺"
rclone copyto "$GDRIVE/3_src/finalize_submission.py" ~/finalize_submission.py || die "finalize 缺"
rclone copy "$GDRIVE/3_src/hsot/" ~/hsot/ --include "*.py" || die "hsot 套件缺"
rclone copyto "$GDRIVE/1_data/drill3_seqs.txt" ~/seqs.txt || die "子集清單缺"
rclone copyto "$GDRIVE/5_outputs/submissions/sub_v049_src_v012_K6.csv" ~/ref_v049.csv || die "參考檔缺"
NSEQ=$(grep -c . ~/seqs.txt); echo "子集 $NSEQ 支"
mark ENV_READY

wait $FC_PID $C3 $C2 || true
grep -q FC_READY ~/fc.log   || die "假色資料未就緒：$(tail -3 ~/fc.log)"
grep -q CKPT3_READY ~/ckpt3.log || die "SAM3 權重未就緒"
grep -q CKPT2_READY ~/ckpt2.log || die "SAM2.1 權重未就緒"
mark DATA_READY

# --- [3] 階段一：e02＝SAMURAI 全量（⚠️ 不傳 --gt-csv）--------------------------
echo "=== [3] e02：SAMURAI SAM2.1-L 全量 $NSEQ 支 ==="
$PY1 ~/track_t1.py --frames-root "$FC" --seq-list ~/seqs.txt --out-dir "$OUT/e02" \
  --backend samurai --samurai-dir "$SAMURAI" --ckpt ~/ckpt/sam2.1_hiera_large.pt \
  || die "e02 失敗"
cp "$OUT/e02/submission.csv" ~/e02.csv
mark E02_DONE; rclone copy "$OUT" "$DEST" --transfers 8 || echo "⚠️ 回傳失敗"

# --- [4] 階段二：e15＝SAM3 全量 ------------------------------------------------
echo "=== [4] e15：SAM3 860M 全量 $NSEQ 支 ==="
$PY3 ~/track_t1.py --frames-root "$FC" --seq-list ~/seqs.txt --out-dir "$OUT/e15" \
  --backend sam3 --sam3-ckpt ~/ckpt/sam3.pt || die "e15 失敗"
cp "$OUT/e15/submission.csv" ~/e15.csv
mark E15_DONE; rclone copy "$OUT" "$DEST" --transfers 8 || echo "⚠️ 回傳失敗"

# --- [5] 階段三：crop prep（只算一次；窗＝e15 ∪ e02）---------------------------
echo "=== [5] crop prep：窗 = e15 ∪ e02、門檻 0.55（D061 定案）==="
PYTHONPATH=~ $PY3 -m hsot.crop_rerun prep --frames-root "$FC" \
  --base-csv ~/e15.csv --envelope-extra ~/e02.csv --area-frac-max 0.55 \
  --out-root ~/crop --meta ~/meta.json | tail -4 || die "crop prep 失敗"
$PY3 -c "
import json,os
m=json.load(open(os.path.expanduser('~/meta.json')))
open(os.path.expanduser('~/crop_seqs.txt'),'w').write('\n'.join(sorted(m))+'\n')
print(f'crop 選中 {len(m)} 支')" || die "crop 清單失敗"
NCROP=$(grep -c . ~/crop_seqs.txt || echo 0)
[ "$NCROP" -ge 1 ] || die "crop 選中 0 支——子集或門檻有問題"
mark CROP_PREP_DONE

# --- [6a] SAM3 在 crop 幀上重跑 → merge base=e15 → v023' ----------------------
echo "=== [6a] SAM3 crop 重跑（$NCROP 支）→ merge base=e15 ==="
$PY3 ~/track_t1.py --frames-root ~/crop --seq-list ~/crop_seqs.txt \
  --out-dir "$OUT/crop_sam3" --backend sam3 --sam3-ckpt ~/ckpt/sam3.pt || die "SAM3 crop 重跑失敗"
PYTHONPATH=~ $PY3 -m hsot.crop_rerun merge --base-csv ~/e15.csv \
  --crop-csv "$OUT/crop_sam3/submission.csv" --meta ~/meta.json \
  --out "$OUT/v023_repro.csv" || die "SAM3 merge 失敗"
mark CROP_SAM3_DONE; rclone copy "$OUT" "$DEST" --transfers 8 || echo "⚠️ 回傳失敗"

# --- [6b] SAMURAI 在【同一組】crop 幀上重跑 → merge base=e02 → v012' -----------
echo "=== [6b] SAMURAI crop 重跑（同窗）→ merge base=e02 ==="
$PY1 ~/track_t1.py --frames-root ~/crop --seq-list ~/crop_seqs.txt \
  --out-dir "$OUT/crop_sam21" --backend samurai --samurai-dir "$SAMURAI" \
  --ckpt ~/ckpt/sam2.1_hiera_large.pt || die "SAMURAI crop 重跑失敗"
PYTHONPATH=~ $PY1 -m hsot.crop_rerun merge --base-csv ~/e02.csv \
  --crop-csv "$OUT/crop_sam21/submission.csv" --meta ~/meta.json \
  --out "$OUT/v012_repro.csv" || die "SAMURAI merge 失敗"
mark CROP_SAM21_DONE; rclone copy "$OUT" "$DEST" --transfers 8 || echo "⚠️ 回傳失敗"

# --- [7] 階段五：finalize（corr both → splice K=6 → validate）-----------------
echo "=== [7] finalize_submission.py ==="
$PY1 ~/finalize_submission.py --main "$OUT/v023_repro.csv" --source "$OUT/v012_repro.csv" \
  --out "$OUT/final.csv" --corr both --K 6 || die "finalize 失敗"
mark FINALIZE_DONE

# --- [8] 驗收：與 v049 的對應列位元級比對 --------------------------------------
echo "=== [8] 驗收：對應列位元級比對 ==="
$PY1 - <<'CHK' > "$OUT/verdict.txt" 2>&1 || { cat "$OUT/verdict.txt"; rclone copy "$OUT" "$DEST" --transfers 8; die "驗收未過"; }
import csv, os, sys, json
H=os.path.expanduser
got={r["ID"]:(r["x"],r["y"],r["width"],r["height"]) for r in csv.DictReader(open(H("~/drill3/final.csv")))}
ref={r["ID"]:(r["x"],r["y"],r["width"],r["height"]) for r in csv.DictReader(open(H("~/ref_v049.csv")))}
seqs=[s.strip() for s in open(H("~/seqs.txt")) if s.strip()]
want={k:v for k,v in ref.items() if k.rsplit("_",1)[0] in seqs}
print(f"子集 {len(seqs)} 支｜產出 {len(got)} 列｜參考對應 {len(want)} 列")
if set(got)!=set(want):
    print(f"🚨 ID 集合不符：缺 {len(set(want)-set(got))}、多 {len(set(got)-set(want))}")
    print("  缺例:", list(set(want)-set(got))[:5]); print("  多例:", list(set(got)-set(want))[:5]); sys.exit(3)
diff=[(k,got[k],want[k]) for k in want if got[k]!=want[k]]
print(f"逐列比對：{len(want)-len(diff)}/{len(want)} 相同、{len(diff)} 不符")
if diff:
    print("🚨 前 5 筆差異：")
    for k,g,w in diff[:5]: print(f"   {k}: got {g}  want {w}")
    # 逐序列彙總，方便定位是哪個階段壞掉
    from collections import Counter
    c=Counter(k.rsplit("_",1)[0] for k,_,_ in diff)
    print("  逐序列不符數：", dict(c.most_common()))
    sys.exit(3)
print("✅ 位元級一致——v049 雙管線交付鏈端到端重現成功")
CHK
cat "$OUT/verdict.txt"

# --- [9] 時間統計（外推 75 支的 Ranking B 預算）--------------------------------
$PY1 - <<'TIM' | tee "$OUT/timing_report.txt"
import os
H=os.path.expanduser
t={}
for line in open(H("~/drill3/timing.txt")):
    k,v=line.split(); t[k]=int(v)
seqs=len([s for s in open(H("~/seqs.txt")) if s.strip()])
order=["DRILL3_START","ENV_READY","DATA_READY","E02_DONE","E15_DONE",
       "CROP_PREP_DONE","CROP_SAM3_DONE","CROP_SAM21_DONE","FINALIZE_DONE"]
prev=None
print(f"{'階段':<20}{'耗時(分)':>10}")
for k in order:
    if k not in t: continue
    if prev: print(f"{k:<20}{(t[k]-t[prev])/60:>10.1f}")
    prev=k
tot=(t[order[-1]]-t[order[0]])/60
gpu=(t["FINALIZE_DONE"]-t["DATA_READY"])/60
print(f"\n總計 {tot:.1f} 分（其中 GPU 階段 {gpu:.1f} 分）｜子集 {seqs} 支")
print(f"⇒ 外推 75 支的 GPU 時間 ≈ {gpu*75/seqs:.0f} 分 = {gpu*75/seqs/60:.1f} 小時")
print(f"  （Ranking B 窗口 9/7–9/10 共 72 小時；環境+資料準備另計 {(t['DATA_READY']-t['DRILL3_START'])/60:.0f} 分）")
TIM

rclone copy "$OUT" "$DEST" --transfers 8 || echo "⚠️ 最終回傳失敗"
echo "✅ 冷啟動演練 #3 完成"
echo "🔻 依 D018 自我 terminate（id=$INSTANCE_ID）"
selfkill
