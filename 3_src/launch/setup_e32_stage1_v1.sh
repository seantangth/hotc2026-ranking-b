#!/usr/bin/env bash
# ============================================================================
# setup_e32_stage0_v1.sh — E32 Stage 1：SAM3 detector 對小目標的 recall 探針
#
# 授權：Sean 08-13 13:34（原核可「$0 本機」，因 SAM3 上游 CUDA-only 改 Lambda，成本變更已核可）
# 判準：5_outputs/strategy_research_20260812/E32_STAGE0_DESIGN_20260813.md（**執行前已寫死**）
#   KILL  跟丟幀偵測率 <30% ⇒ SAM3.1 的 3/98 警訊重現 ⇒ E32 整條線判死，不進 Stage 1
#   GO    top-1 IoU≥0.5 命中率 ≥40% ⇒ 進 Stage 1
#   GREY  其餘 ⇒ 不自動花錢，呈 Sean 裁決
#
# 【為何不能在本機跑】sam3 推論路徑 30 處硬編 device="cuda"／.cuda() ＋ 11 處 autocast("cuda")
#   ⇒ 逐一 patch 會改變語意（D039/D062 風險）且與生產環境不同源。判決見 DESIGN 檔末節。
#
# 🚨 terminate 一律用傳入的 INSTANCE_ID（Sean 08-12 指示：只關自己開的機器）
# 【昨日兩次夭折的教訓，本腳本已內建】①新機沒有 rclone，第一個 rclone 呼叫前必裝（D067(f)④）
#   ②環境依賴清單整段抄 drill v2 的 sam3env，不抄一半（漏 loguru 害死二號機）
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

INSTANCE_ID="${INSTANCE_ID:?必須傳入 INSTANCE_ID}"
GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DEST="$GDRIVE/5_outputs/e32_stage1_20260813"
SAM3_SHA=96914d2425f90a64f45ca977c2b5165418099543   # D059⑤ 釘死，與 v008/v023 生產同源
PY3=~/sam3env/bin/python
export E32_DIR=~/e32
OUT=~/e32_out; mkdir -p "$OUT" "$E32_DIR"

selfkill() {
  key=$(cat ~/.lambda_key | tr -d '\n'); [ -z "$key" ] && return 0
  curl -s -u "$key:" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H 'Content-Type: application/json' -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}"
}
die() {
  echo "🚨 $*"
  if command -v rclone >/dev/null 2>&1; then
    rclone copy "$OUT" "$DEST/out" --transfers 8 2>/dev/null
    echo "DIED: $*" > /tmp/e32_died.txt
    rclone copyto /tmp/e32_died.txt "$DEST/DIED.txt" 2>/dev/null
  fi
  selfkill; exit 1
}

# --- [0] 保險自毀（90 分；本作業預估 25 分）------------------------------------
pgrep -af "sleep [0-9]+" || echo "（無既有計時器）"
nohup bash -c "sleep 10800
  key=\$(cat ~/.lambda_key | tr -d '\n'); [ -z \"\$key\" ] && exit 0
  curl -s -u \"\$key:\" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H 'Content-Type: application/json' -d '{\"instance_ids\":[\"$INSTANCE_ID\"]}'" >/dev/null 2>&1 &
echo "保險自毀已掛（180 分）"

# --- [0a] rclone（新機沒有，D067(f)④）------------------------------------------
if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | sudo bash >/dev/null 2>&1 || die "rclone 安裝失敗"
fi
rclone lsf "$GDRIVE/1_data/" --max-depth 1 >/dev/null 2>&1 || die "rclone 讀不到 gDrive"
echo "rclone OK"

# --- [1] 資料與腳本（單 tar，D036）---------------------------------------------
( set -e
  rclone copyto "$GDRIVE/1_data/packed/t1val_fc_65.tar" ~/val_fc.tar
  mkdir -p ~/valfc && tar -xf ~/val_fc.tar -C ~/valfc && echo VALFC_READY
) > ~/pull_fc.log 2>&1 &
PULL=$!
rclone copyto "$GDRIVE/3_src/peft/stage1_probe_cuda.py" "$E32_DIR/stage1_probe.py" || die "probe 腳本缺"
rclone copyto "$GDRIVE/3_src/peft/stage1_frames.json" "$E32_DIR/stage1_frames.json" || die "樣本清單缺"

# --- [2] sam3env（整段抄 drill v2，勿刪任何一項）-------------------------------
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -d ~/sam3env ] || uv venv --python 3.12 ~/sam3env
if [ ! -f ~/sam3env/.deps_done ]; then
  VIRTUAL_ENV=~/sam3env uv pip install -q torch torchvision --torch-backend=auto || die "torch 失敗"
  VIRTUAL_ENV=~/sam3env uv pip install -q "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}" \
    "numpy<2" ftfy==6.1.1 regex einops psutil scipy av \
    pycocotools numba python-rapidjson pandas pillow tqdm opencv-python-headless || die "sam3 失敗"
  VIRTUAL_ENV=~/sam3env uv pip install -q "setuptools<81" huggingface_hub || die "附屬失敗"
  touch ~/sam3env/.deps_done
fi
$PY3 -c "
import sam3, numpy, torch, pkg_resources
assert numpy.__version__.startswith('1.'), f'numpy {numpy.__version__} 應 <2'
assert torch.cuda.is_available(), 'CUDA 不可用'
print(f'sam3env OK: numpy {numpy.__version__} | {torch.cuda.get_device_name(0)}')
" 2>&1 | grep -viE 'warning|deprecat|^ +import' || die "sam3env 驗證失敗"

# --- [3] SAM3 權重（與 v008/v023 同源鏡像；官方 facebook/sam3 檔案層仍 gated）----
$PY3 -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('1038lab/sam3','sam3.pt', local_dir='$E32_DIR/ckpt')
print('ckpt:', p)
" || die "權重下載失敗"

# --- [4] 幀就位 -----------------------------------------------------------------
wait $PULL; grep -q VALFC_READY ~/pull_fc.log || die "val_fc tar 未就緒"
$PY3 - <<'PYX' || die "幀樹建立失敗"
import json, os, pathlib
E = pathlib.Path(os.environ["E32_DIR"]); root = pathlib.Path.home()/"valfc"
seqs = list(json.load(open(E/"stage1_frames.json"))["seqs"])
(E/"frames").mkdir(parents=True, exist_ok=True)
miss = []
for s in seqs:
    src = next((p for p in root.rglob(s) if p.is_dir()), None)
    if src is None: miss.append(s); continue
    dst = E/"frames"/s
    if dst.is_symlink() or dst.exists(): dst.unlink()
    dst.symlink_to(src)
print(f"幀樹：{len(seqs)-len(miss)}/{len(seqs)} 支就位" + (f"；缺 {miss}" if miss else ""))
assert not miss, f"缺序列 {miss}"
PYX

# --- [5] 探針（判準寫在 DESIGN 檔與腳本內，此處僅執行）--------------------------
$PY3 "$E32_DIR/stage1_probe.py" 2>&1 | tee "$OUT/stage1_console.log"
RC=${PIPESTATUS[0]}
cp "$E32_DIR/stage1_result.json" "$OUT/" 2>/dev/null
rclone copy "$OUT" "$DEST/out" --transfers 8 || echo "⚠️ 回傳失敗"
[ $RC -ne 0 ] && echo "⚠️ 探針非零退出 rc=$RC（log 已回傳，可本機重判）"

echo "✅ E32 Stage 1 完成"; tail -5 "$OUT/stage1_console.log"
echo "🔻 依 D018 自我 terminate（id=$INSTANCE_ID）"
selfkill
