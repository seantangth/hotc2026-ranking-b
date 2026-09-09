#!/usr/bin/env bash
# ============================================================================
# setup_e48_disagree_v1.sh — E48 分歧閘 exemplar 重偵測 canary
#
# 判準：5_outputs/strategy_research_20260812/E48_DISAGREE_GATE_DESIGN_20260821.md
# 授權：Sean 08-21「上啊」。零 LB。結束 terminate $INSTANCE_ID（D018）。
# SAM3 SHA 釘 96914d24…（D059）。不掃帳號、不 terminate 別台。
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

INSTANCE_ID="${INSTANCE_ID:?必須傳入 INSTANCE_ID}"
GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DEST="$GDRIVE/5_outputs/e48_disagree_20260821"
SAM3_SHA=96914d2425f90a64f45ca977c2b5165418099543
PY3=~/sam3env/bin/python
export E48_DIR=~/e48
OUT=~/e48_out
mkdir -p "$OUT" "$E48_DIR"

selfkill() {
  key=$(tr -d '\n' < ~/.lambda_key 2>/dev/null || true)
  [ -z "$key" ] && return 0
  curl -s -u "$key:" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H 'Content-Type: application/json' -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}"
}
die() {
  echo "🚨 $*"
  if command -v rclone >/dev/null 2>&1; then
    rclone copy "$OUT" "$DEST/out" --transfers 8 2>/dev/null || true
    echo "DIED: $*" > /tmp/e48_died.txt
    rclone copyto /tmp/e48_died.txt "$DEST/DIED.txt" 2>/dev/null || true
  fi
  selfkill
  exit 1
}

pgrep -af "sleep [0-9]+" || echo "（無既有計時器）"
nohup bash -c "sleep 10800
  key=\$(tr -d '\n' < ~/.lambda_key 2>/dev/null || true); [ -z \"\$key\" ] && exit 0
  curl -s -u \"\$key:\" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H 'Content-Type: application/json' -d '{\"instance_ids\":[\"$INSTANCE_ID\"]}'" >/dev/null 2>&1 &
echo "保險自毀已掛（180 分）"

if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | sudo bash >/dev/null 2>&1 || die "rclone 安裝失敗"
fi
[ -s ~/.config/rclone/rclone.conf ] || die "rclone.conf 未上傳"
rclone lsf "$GDRIVE/1_data/" --max-depth 1 >/dev/null 2>&1 || die "rclone 讀不到 gDrive"
echo "rclone OK"

( set -e
  rclone copyto "$GDRIVE/1_data/packed/t1val_fc_65.tar" ~/val_fc.tar
  mkdir -p ~/valfc && tar -xf ~/val_fc.tar -C ~/valfc && echo VALFC_READY
) > ~/pull_fc.log 2>&1 &
PULL=$!
rclone copyto "$GDRIVE/3_src/peft/e48_probe_cuda.py" "$E48_DIR/e48_probe.py" || die "probe 缺"
rclone copyto "$GDRIVE/3_src/peft/e48_frames.json" "$E48_DIR/e48_frames.json" || die "frames json 缺"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
[ -d ~/sam3env ] || uv venv --python 3.12 ~/sam3env
if [ ! -f ~/sam3env/.deps_done ]; then
  VIRTUAL_ENV=~/sam3env uv pip install -q torch torchvision --torch-backend=auto || die "torch 失敗"
  VIRTUAL_ENV=~/sam3env uv pip install -q "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}" \
    "numpy<2" ftfy==6.1.1 regex einops psutil scipy av \
    pycocotools numba python-rapidjson pandas pillow tqdm opencv-python-headless || die "sam3 失敗"
  VIRTUAL_ENV=~/sam3env uv pip install -q "setuptools<81" huggingface_hub || die "附屬失敗"
  touch ~/sam3env/.deps_done || die "deps_done 標記失敗"
fi
$PY3 -c "
import numpy, torch
assert numpy.__version__.startswith('1.'), numpy.__version__
assert torch.cuda.is_available(), 'CUDA 不可用'
print('sam3env OK', numpy.__version__, torch.cuda.get_device_name(0))
" || die "sam3env 驗證失敗"

$PY3 -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('1038lab/sam3','sam3.pt', local_dir='$E48_DIR/ckpt')
print('ckpt:', p)
" || die "權重下載失敗"

wait $PULL
grep -q VALFC_READY ~/pull_fc.log || die "val_fc tar 未就緒"
$PY3 - <<'PYX' || die "幀樹建立失敗"
import json, os, pathlib
E = pathlib.Path(os.environ["E48_DIR"]); root = pathlib.Path.home()/"valfc"
seqs = list(json.load(open(E/"e48_frames.json"))["seqs"])
(E/"frames").mkdir(parents=True, exist_ok=True)
miss = []
for s in seqs:
    src = next((p for p in root.rglob(s) if p.is_dir()), None)
    if src is None:
        miss.append(s); continue
    dst = E/"frames"/s
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)
print(f"幀樹：{len(seqs)-len(miss)}/{len(seqs)}" + (f" 缺 {miss}" if miss else ""))
assert not miss, miss
PYX

$PY3 "$E48_DIR/e48_probe.py" 2>&1 | tee "$OUT/e48_console.log"
RC=${PIPESTATUS[0]}
cp "$E48_DIR/e48_result.json" "$OUT/" 2>/dev/null || true
# 精簡回傳：records 很大，另存無 records 的摘要
$PY3 - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["E48_DIR"])/"e48_result.json"
if p.exists():
    d = json.loads(p.read_text())
    d.pop("records", None)
    Path.home().joinpath("e48_out/e48_summary.json").write_text(json.dumps(d, indent=1, ensure_ascii=False))
    print("VERDICT", d.get("verdict"), "gates", d.get("gates"))
PY
rclone copy "$OUT" "$DEST/out" --transfers 8 || echo "⚠️ 回傳失敗"
[ "$RC" -ne 0 ] && echo "⚠️ 探針 rc=$RC（log 已回傳）"
echo "✅ E48 完成"; tail -20 "$OUT/e48_console.log"
echo "🔻 terminate id=$INSTANCE_ID"
selfkill
