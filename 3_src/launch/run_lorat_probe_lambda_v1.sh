#!/usr/bin/env bash
# LoRAT zero-shot 探針（09-04）：現代 SOT tracker 裸機打不打得贏 SAM3？
#
# 背景：08-03 的候選表列了 LoRAT（DINOv2 ＋ LoRA、Apache-2.0），隨後整條 T2 線被 D028
# 的 ViPT 失敗連坐關閉，LoRAT **從未被測過**。08-26 稽核把「從 n=1 推論整個 SOT 類別」
# 列為方法瑕疵。本探針補這一格，成本 ~$4。
#
# 判準（事前寫死，跑前不得改；對照＝同一份 val65 的既有讀數）：
#   E02 SAMURAI 0.68985 ／ E15 SAM3 0.68843 ／ E46 crop-SAM3 0.70039
#   最佳 LoRAT 變體的 val65 pooled AUC：
#     ≥ 0.72  ⇒ 底座選錯，呈 Sean 緊急裁決（36 小時內能否換軌）
#     0.69–0.72 ⇒ 可比但不佔優；納入「多來源拼接」候選，不換軌
#     < 0.69  ⇒ 現代 SOT tracker 在假色高光譜上不優於 SAM3 ⇒ 底座選擇確認正確，此軸關閉
#   ⚠️ 三個分支覆蓋完整結果空間（08-30 制度項）。
#
# 五階段，逐階 fail-closed，**模型建構與權重載入排在資料之前**（早失敗，省機時）：
#   0 環境 ＋ LoRAT clone ＋ 權重（gdown）＋ val65 資料（gDrive tar）
#   1 模型建構 ＋ 權重載入自檢（缺鍵即 die——D051 SAM 3.1 就是這個失敗型態）
#   2 兩支序列冒煙
#   3 L-224 全 val65 → hsot.eval 計分
#   4 g-378 全 val65 → hsot.eval 計分 → summary
#
# ── SELFKILL 算術 ──────────────────────────────────────────────────────────
#   env 8 ＋ 權重下載 15（large 1.2GB ＋ giant-378 4.5GB，Google Drive 速度未知）
#   ＋ 資料 5 ＋ 自檢 3 ＋ 冒煙 3 ＋ L-224 全量 20 ＋ g-378 全量 60 ＋ 收尾 6 ≈ 120 分
#   ⇒ rearm 200（×1.65 緩衝，吸收 Google Drive 限速）
set -euo pipefail

export PYTHONUNBUFFERED=1
export PATH="/home/ubuntu/.local/bin:$PATH"
export RCLONE_CONFIG=/home/ubuntu/rclone.conf

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DEST="$GDRIVE/5_outputs/lorat_probe_20260904"
LORAT_SHA=5260744ff0a65207a73289bbda1788c377d23bcc

die() { echo "FATAL: $*" >&2; exit 1; }

sudo test -x /root/rearm_selfkill.sh \
  || die "缺 /root/rearm_selfkill.sh：cloud-init selfkill backstop 未就位"

# GPU 閘門（08-31 壞卡事故：ECC 未修正錯誤會讓推論半途 SIGABRT，極易誤判成程式問題）
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
ECC="$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader | head -1 | tr -d ' ')"
case "$ECC" in
  0|"[N/A]"|"N/A"|"") ;;
  *) die "GPU ECC uncorrected=$ECC（壞卡，terminate 換機）";;
esac
echo "GPU=$GPU_NAME ECC=$ECC"
sudo /root/rearm_selfkill.sh 200

REPO=/home/ubuntu/hsot_repo        # D093：不可叫 hsot（與套件同名撞 namespace）
SRC="$REPO/3_src"
WORK=/home/ubuntu/lorat_probe
LORAT=/home/ubuntu/LoRAT
FRAMES=/home/ubuntu/val65_fc
CKPT=/home/ubuntu/lorat_ckpt
ENV=/home/ubuntu/loratenv
PY="$ENV/bin/python"
SYNC_PID=""

test -d "$SRC" || die "repo 不在 $REPO（開機流程應先放好）"
export PYTHONPATH="$SRC"           # D093／08-31：獨立腳本必須自己帶 PYTHONPATH

sync_progress() {
  while true; do
    rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 --exclude '*.part' || true
    sleep 60
  done
}
finish() {
  RUN_RC=$?
  trap - EXIT; set +e
  if [ -n "$SYNC_PID" ]; then kill "$SYNC_PID" 2>/dev/null; wait "$SYNC_PID" 2>/dev/null; fi
  echo "$(date -Is) finish rc=$RUN_RC" | tee -a "$WORK/finish.txt"
  SYNC_RC=0
  if command -v rclone >/dev/null 2>&1 && [ -f "$RCLONE_CONFIG" ]; then
    rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 --exclude '*.part' || SYNC_RC=$?
    [ "$SYNC_RC" -eq 0 ] && { rclone check "$WORK" "$DEST" --one-way --exclude '*.part' || SYNC_RC=$?; }
  else
    SYNC_RC=127
  fi
  if [ "$SYNC_RC" -eq 0 ]; then
    sudo /root/rearm_selfkill.sh 5
  else
    echo "$(date -Is) REMOTE_VERIFY_FAILED rc=$SYNC_RC; selfkill 未縮短" | tee -a "$WORK/finish.txt"
    [ "$RUN_RC" -eq 0 ] && RUN_RC=3
  fi
  exit "$RUN_RC"
}
trap finish EXIT
trap 'echo "死於第 $LINENO 行" >&2' ERR

mkdir -p "$WORK/logs" "$FRAMES" "$CKPT"
chmod 600 "$RCLONE_CONFIG"
echo "$(date -Is) SETUP_START gpu=$GPU_NAME" | tee "$WORK/timeline.txt"

command -v rclone >/dev/null 2>&1 || curl -fsSL https://rclone.org/install.sh | sudo bash >/dev/null
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
rclone lsf "$GDRIVE/1_data/packed" >/dev/null

# ── 階段 0a：資料與權重（背景並行）────────────────────────────────────────
(
  set -euo pipefail
  rclone copyto "$GDRIVE/1_data/packed/t1val_fc_65.tar" /home/ubuntu/val_fc_65.tar
  tar -xf /home/ubuntu/val_fc_65.tar -C "$FRAMES"
  test "$(find "$FRAMES" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 65
  rclone copyto "$GDRIVE/1_data/raw/2026training.csv" /home/ubuntu/2026training.csv
  test "$(wc -l < /home/ubuntu/2026training.csv)" -eq 169491
  echo DATA_READY
) > "$WORK/logs/data.log" 2>&1 &
DATA_PID=$!

# ── 階段 0b：環境 ＋ LoRAT ────────────────────────────────────────────────
[ -x "$PY" ] || uv venv --python 3.12 "$ENV"
VIRTUAL_ENV="$ENV" uv pip install -q torch torchvision --torch-backend=auto
VIRTUAL_ENV="$ENV" uv pip install -q timm safetensors pillow numpy pandas tqdm
"$PY" -c "import torch,timm,safetensors; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

[ -d "$LORAT/.git" ] || git clone -q https://github.com/LitingLin/LoRAT.git "$LORAT"
git -C "$LORAT" fetch -q --depth 1 origin "$LORAT_SHA"
git -C "$LORAT" checkout -q --detach "$LORAT_SHA"
grep -q "Apache License" "$LORAT/LICENSE" || die "LoRAT LICENSE 非 Apache（授權前提變了，停）"

# 權重（Google Drive；官方唯一發布管道）。GOT10k/ 子目錄那批是 GOT-10k-only 協定版，不取。
VIRTUAL_ENV="$ENV" uv pip install -q gdown
"$PY" -m gdown --no-cookies -O "$CKPT/large.bin"      1joL2VmWxrIhkapj7HykoTLYKmLiBgXqm
"$PY" -m gdown --no-cookies -O "$CKPT/giant-378.bin"  1nYrKT4EfvqkSxUMqcCCBimgIsfakk7YL
ls -la "$CKPT" | tee "$WORK/weights_listing.txt"
# ⚠️ 檔案小是**正常的**：官方 .bin 只含 LoRA 增量＋head，不含 backbone
# （large 130MB／32.5M 參數、giant-378 321MB）。門檻只用來擋 gdown 抓到 HTML 配額頁。
test "$(stat -c%s "$CKPT/large.bin")"     -gt 100000000 || die "large.bin 太小（gdown 可能抓到 HTML 配額頁）"
test "$(stat -c%s "$CKPT/giant-378.bin")" -gt 250000000 || die "giant-378.bin 太小（同上）"
head -c 8 "$CKPT/large.bin" | grep -q . && \
  "$PY" -c "import sys; from safetensors.torch import load_file; load_file(sys.argv[1]) and None" "$CKPT/large.bin" \
  || die "large.bin 不是合法 safetensors（副檔名雖為 .bin，格式是 safetensors）"
sha256sum "$CKPT"/*.bin > "$WORK/checkpoint_sha256.txt"

wait "$DATA_PID"; grep -q DATA_READY "$WORK/logs/data.log"
uv pip freeze --python "$PY" > "$WORK/lorat_freeze.txt"
nvidia-smi > "$WORK/nvidia_smi.txt" || true
echo "$(date -Is) ENV_DATA_OK" | tee -a "$WORK/timeline.txt"

sync_progress & SYNC_PID=$!

# ── 階段 1：模型建構 ＋ 權重載入自檢（在碰資料之前；缺鍵即 die）──────────
for V in L-224:large.bin g-378:giant-378.bin; do
  VAR="${V%%:*}"; WFILE="${V##*:}"
  "$PY" - "$LORAT" "$SRC" "$VAR" "$CKPT/$WFILE" <<'PYEOF'
import sys
lorat_root, src_root, variant, weight = sys.argv[1:5]
sys.path.insert(0, lorat_root); sys.path.insert(0, src_root + "/lorat_probe")
import torch
from run_lorat_val65 import build_model
model, rep = build_model(variant, weight, torch.device("cpu"), torch.float32)
n = sum(p.numel() for p in model.parameters())
print(f"[selfcheck] {variant}: params={n/1e6:.1f}M missing={rep['n_missing']} unexpected={rep['n_unexpected']}")
PYEOF
done 2>&1 | tee "$WORK/logs/selfcheck.log"
echo "$(date -Is) SELFCHECK_OK" | tee -a "$WORK/timeline.txt"

# ── 階段 2：兩支序列冒煙（跑得動 ＋ 分數不是 0）──────────────────────────
"$PY" "$SRC/lorat_probe/run_lorat_val65.py" \
  --lorat-root "$LORAT" --variant L-224 --weight "$CKPT/large.bin" \
  --frames-root "$FRAMES" --gt-csv /home/ubuntu/2026training.csv \
  --seqs "$REPO/1_data/val_split_v1.txt" --out "$WORK/smoke_L224.csv" \
  --limit-seqs 2 2>&1 | tee "$WORK/logs/smoke.log"
"$PY" -m hsot.eval "$WORK/smoke_L224.csv" /home/ubuntu/2026training.csv \
  2>&1 | tee "$WORK/logs/smoke_eval.log"
echo "$(date -Is) SMOKE_OK" | tee -a "$WORK/timeline.txt"

# ── 階段 3+4：兩個變體全 val65 ＋ 計分 ────────────────────────────────────
for V in L-224:large.bin g-378:giant-378.bin; do
  VAR="${V%%:*}"; WFILE="${V##*:}"
  echo "$(date -Is) RUN_START $VAR" | tee -a "$WORK/timeline.txt"
  "$PY" "$SRC/lorat_probe/run_lorat_val65.py" \
    --lorat-root "$LORAT" --variant "$VAR" --weight "$CKPT/$WFILE" \
    --frames-root "$FRAMES" --gt-csv /home/ubuntu/2026training.csv \
    --seqs "$REPO/1_data/val_split_v1.txt" \
    --out "$WORK/sub_lorat_${VAR}_val65.csv" 2>&1 | tee "$WORK/logs/run_${VAR}.log"
  "$PY" -m hsot.eval "$WORK/sub_lorat_${VAR}_val65.csv" /home/ubuntu/2026training.csv \
    --seqs "$REPO/1_data/val_split_v1.txt" --per-seq \
    2>&1 | tee "$WORK/logs/eval_${VAR}.log"
  echo "$(date -Is) RUN_DONE $VAR" | tee -a "$WORK/timeline.txt"
done

# ── 匯總（判準在此機械套用，不留給人腦）──────────────────────────────────
"$PY" - "$WORK" <<'PYEOF'
import json, re, sys
from pathlib import Path
work = Path(sys.argv[1])
BASELINES = {"E02_samurai_val65": 0.68985, "E15_sam3_val65": 0.68843, "E46_crop_sam3_val65": 0.70039}
res = {}
for p in sorted(work.glob("logs/eval_*.log")):
    variant = p.stem.replace("eval_", "")
    m = re.search(r"pooled\s+AUC=([0-9.]+)", p.read_text())
    if m:
        res[variant] = float(m.group(1))
best_v, best = (max(res.items(), key=lambda kv: kv[1]) if res else (None, None))
if best is None:
    verdict = "ERROR_NO_SCORE"
elif best >= 0.72:
    verdict = "BASE_CHOICE_WRONG_ESCALATE"
elif best >= 0.69:
    verdict = "COMPARABLE_NOT_SUPERIOR"
else:
    verdict = "SAM3_CONFIRMED_BETTER"
doc = {"val65_pooled_auc": res, "best_variant": best_v, "best_auc": best,
       "baselines": BASELINES,
       "delta_vs_sam3": (best - BASELINES["E15_sam3_val65"]) if best else None,
       "delta_vs_crop_sam3": (best - BASELINES["E46_crop_sam3_val65"]) if best else None,
       "verdict": verdict,
       "criteria": {">=0.72": "BASE_CHOICE_WRONG_ESCALATE",
                    "0.69-0.72": "COMPARABLE_NOT_SUPERIOR", "<0.69": "SAM3_CONFIRMED_BETTER"}}
(work / "lorat_verdict.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))
print(json.dumps(doc, ensure_ascii=False, indent=2))
PYEOF
test -s "$WORK/lorat_verdict.json"
echo "$(date -Is) PROBE_DONE" | tee -a "$WORK/timeline.txt"
cat "$WORK/lorat_verdict.json"
