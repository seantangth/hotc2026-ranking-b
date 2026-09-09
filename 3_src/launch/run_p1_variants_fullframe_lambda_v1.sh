#!/usr/bin/env bash
# P1 decoder 修法變體的 full-frame 篩選（2026-09-06 上午，Sean 09:42「趕緊開始補救」；依 TECHNICAL_ROUTE_REVIEW §六）：
#   以 frozen sam3.pt ＋ P1 arm A decoder ckpt 造三個變體，各跑 test75 full-frame SAM3 腿，產物邊跑邊回 gDrive，
#   本機以 v096 式組裝（只換 43 支未裁切序列的 main ＋ 第三腿）打 LB 篩選。事前判準見 EXPERIMENT_LOG 09-06 節。
#   變體：heads_frozen（mask decoder 用 P1、iou_prediction_head 與 pred_obj_score_head 還原 frozen）／alpha050／alpha025（decoder 權重線性插值）。
# 只需 sam3env（不裝 SAMURAI／t1env）。⏰ 自毀：cloud-init 唯一計時器；A100 rearm 180（環境 10 ＋ 3 腿 × ~30 ＋ 收尾）。
set -euo pipefail
trap 'echo "死於第 $LINENO 行" >&2' ERR
export PYTHONUNBUFFERED=1
export PATH="/home/ubuntu/.local/bin:$PATH"
export RCLONE_CONFIG=/home/ubuntu/rclone.conf
GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
RUN_NAME="${RUN_NAME:-p1_variants_fullframe_20260906}"
DEST="$GDRIVE/5_outputs/$RUN_NAME"
P1_SD="$GDRIVE/5_outputs/p1_precision_train_20260902/run/tracker_p1_A_decoder_best.pt"
REPO=/home/ubuntu/hsot_repo; SRC="$REPO/3_src"; WORK="/home/ubuntu/$RUN_NAME"
FRAMES=/home/ubuntu/test_fc; CKPT=/home/ubuntu/ckpt; SAM3ENV=/home/ubuntu/sam3env; PY3="$SAM3ENV/bin/python"
SAM3_SHA=96914d2425f90a64f45ca977c2b5165418099543
SAM3_CKPT_SHA256=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
SAMPLE=/home/ubuntu/sample_submisson.csv
PINNED_ENV="${PINNED_ENV:-1}"
VARIANTS="${VARIANTS:-heads_frozen,alpha050,alpha025}"
SYNC_PID=""
die() { echo "FATAL: $*" >&2; exit 1; }
: "${HF_TOKEN:?HF_TOKEN is required}"
sudo test -x /root/rearm_selfkill.sh || die "缺 cloud-init selfkill backstop"
test -d "$SRC" || die "repo 不在 $REPO"
export PYTHONPATH="$SRC"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
ECC=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d " ")
case "$ECC" in 0|"[N/A]"|"N/A"|"") echo "GPU_ECC_OK gpu=$GPU_NAME";; *) die "GPU ECC uncorrected=$ECC 壞卡，換機";; esac
if [[ "$GPU_NAME" == *A100* || "$GPU_NAME" == *H100* ]]; then SELFKILL_MIN="${REARM_MIN:-180}"; else SELFKILL_MIN="${REARM_MIN:-300}"; fi
sudo /root/rearm_selfkill.sh "$SELFKILL_MIN"
sync_progress() { while true; do rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 --exclude '*.part' --exclude '*.part/**' || true; sleep 60; done; }
finish() {
  RUN_RC=$?; trap - EXIT; set +e
  [ -n "$SYNC_PID" ] && { kill "$SYNC_PID" 2>/dev/null; wait "$SYNC_PID" 2>/dev/null; }
  echo "$(date -Is) finish rc=$RUN_RC" | tee -a "$WORK/finish.txt"
  SYNC_RC=0
  rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 --exclude '*.part' --exclude '*.part/**' || SYNC_RC=$?
  [ "$SYNC_RC" -eq 0 ] && { rclone check "$WORK" "$DEST" --one-way --exclude '*.part' --exclude '*.part/**' || SYNC_RC=$?; }
  if [ "$SYNC_RC" -eq 0 ]; then sudo /root/rearm_selfkill.sh 5; else echo "REMOTE_VERIFY_FAILED rc=$SYNC_RC" | tee -a "$WORK/finish.txt"; [ "$RUN_RC" -eq 0 ] && RUN_RC=3; fi
  exit "$RUN_RC"
}
trap finish EXIT
mkdir -p "$WORK/logs" "$WORK/variants" "$CKPT" "$FRAMES"; chmod 600 "$RCLONE_CONFIG"
echo "$(date -Is) SETUP_START gpu=$GPU_NAME variants=$VARIANTS" | tee "$WORK/timeline.txt"
command -v rclone >/dev/null 2>&1 || curl -fsSL https://rclone.org/install.sh | sudo bash >/dev/null
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
( set -euo pipefail
  rclone copyto "$GDRIVE/1_data/packed/t1test_fc_75.tar" /home/ubuntu/test_fc_75.tar
  tar -xf /home/ubuntu/test_fc_75.tar -C "$FRAMES"; test "$(find "$FRAMES" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 75
  rclone copyto "$GDRIVE/1_data/raw/sample_submisson.csv" "$SAMPLE"; test -s "$SAMPLE"
  rclone copyto "$P1_SD" "$CKPT/tracker_p1_A_decoder_best.pt"; test "$(stat -c%s "$CKPT/tracker_p1_A_decoder_best.pt")" -eq 47086331
  if rclone copyto "$GDRIVE/4_models/pretrained/sam3.pt" "$CKPT/sam3.pt" 2>/dev/null && echo "$SAM3_CKPT_SHA256  $CKPT/sam3.pt" | sha256sum -c - >/dev/null 2>&1; then echo gdrive > "$WORK/SAM3_WEIGHT_SOURCE.txt"
  else rm -f "$CKPT/sam3.pt"; curl -fL -H "Authorization: Bearer $HF_TOKEN" -o "$CKPT/sam3.pt" "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt"; echo "$SAM3_CKPT_SHA256  $CKPT/sam3.pt" | sha256sum -c -; echo hf > "$WORK/SAM3_WEIGHT_SOURCE.txt"; fi
  echo DATA_READY ) > "$WORK/logs/data_setup.log" 2>&1 &
DATA_PID=$!
[ -x "$SAM3ENV/bin/python" ] || uv venv --python 3.12 "$SAM3ENV"
VIRTUAL_ENV="$SAM3ENV" uv pip install -q torch torchvision --torch-backend=auto
if [ "$PINNED_ENV" = 1 ]; then
  VIRTUAL_ENV="$SAM3ENV" uv pip install -q -r "$SRC/requirements-sam3env.txt"
  VIRTUAL_ENV="$SAM3ENV" uv pip install -q "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}"
else
  VIRTUAL_ENV="$SAM3ENV" uv pip install -q "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}" "numpy<2" ftfy==6.1.1 regex einops psutil scipy av pycocotools numba python-rapidjson pandas pillow tqdm opencv-python-headless
fi
VIRTUAL_ENV="$SAM3ENV" uv pip install -q "setuptools<81"
"$PY3" -c "import torch,sam3,numpy; assert torch.cuda.is_available(); assert numpy.__version__.startswith('1.'); print(torch.__version__, torch.cuda.get_device_name(0))"
wait "$DATA_PID"; grep -q DATA_READY "$WORK/logs/data_setup.log"
sha256sum "$CKPT/sam3.pt" "$CKPT/tracker_p1_A_decoder_best.pt" > "$WORK/checkpoint_sha256.txt"
uv pip freeze --python "$PY3" > "$WORK/sam3_freeze.txt"
echo "$(date -Is) ENV_DATA_OK" | tee -a "$WORK/timeline.txt"
sync_progress & SYNC_PID=$!

# ── 造變體（沿用 merge_ckpt 的載入／同構檢查；輸出不加 metadata 鍵，provenance 寫 sidecar）──
"$PY3" - "$CKPT/sam3.pt" "$CKPT/tracker_p1_A_decoder_best.pt" "$WORK/variants" "$VARIANTS" 2>&1 <<'PYEOF' | tee "$WORK/logs/build_variants.log"
import sys, json, hashlib, torch
from pathlib import Path
sys.path.insert(0, "/home/ubuntu/hsot_repo/3_src")
from train_tracker_g1.merge_ckpt import load_full_ckpt, load_tracker_sd, TRACKER_PREFIX
base_p, sd_p, out_dir, variants = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4].split(",")
base, wrapped = load_full_ckpt(base_p); sd, meta = load_tracker_sd(sd_p)
p1 = {TRACKER_PREFIX + k: v for k, v in sd.items()}
missing = [k for k in p1 if k not in base]; assert not missing, missing[:5]
changed = [k for k in p1 if not torch.equal(p1[k].to(base[k].dtype), base[k])]
heads = [k for k in p1 if ("iou_prediction_head" in k or "pred_obj_score_head" in k)]
print(f"p1 keys={len(p1)} changed_vs_base={len(changed)} head_keys={len(heads)} changed_heads={sum(1 for k in heads if k in changed)}")
print("changed 前 8:", changed[:8]); print("heads 前 6:", heads[:6])
assert changed, "P1 sd 與 base 完全相同？"
def save(name, merged):
    out = out_dir / f"merged_{name}.pt"; torch.save({"model": merged} if wrapped else merged, out)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    (out_dir / f"merged_{name}.provenance.json").write_text(json.dumps({"variant": name, "base": base_p, "p1_sd": sd_p, "sha256": h, "n_changed_keys": sum(1 for k in p1 if not torch.equal(merged[k], base[k]))}, indent=1))
    print(f"[{name}] written sha256={h[:16]} size={out.stat().st_size}")
for name in variants:
    m = dict(base)
    if name == "heads_frozen":
        for k, v in p1.items():
            m[k] = base[k] if k in heads else v.to(base[k].dtype)
    elif name.startswith("alpha"):
        a = int(name[5:]) / 100.0
        for k, v in p1.items():
            if torch.is_floating_point(base[k]):
                m[k] = ((1 - a) * base[k].float() + a * v.float()).to(base[k].dtype)
            else:
                m[k] = v.to(base[k].dtype)
    else:
        raise SystemExit(f"未知變體 {name}")
    save(name, m)
PYEOF
echo "$(date -Is) VARIANTS_BUILT" | tee -a "$WORK/timeline.txt"

# ── 各變體 full-frame 腿（旗標同 run_ranking_b 的 full primary）─────────────────
IFS=',' read -r -a VLIST <<< "$VARIANTS"
for V in "${VLIST[@]}"; do
  echo "$(date -Is) LEG_START $V" | tee -a "$WORK/timeline.txt"
  "$PY3" "$SRC/track_t1.py" --frames-root "$FRAMES" --out-dir "$WORK/leg_$V" --backend sam3 \
    --sample-csv "$SAMPLE" --sam3-ckpt "$WORK/variants/merged_$V.pt" --sam3-eval \
    --source-revision "$SAM3_SHA" 2>&1 | tee "$WORK/logs/leg_$V.log" | grep -vE '^\s*[0-9]+%\|' || true
  test -s "$WORK/leg_$V/submission.csv" || die "leg $V 無輸出"
  "$PY3" -c "import json,sys; d=json.load(open(sys.argv[1])); m=d['_meta']; assert m.get('validation')=='pass', m; print('leg $V validation pass, n_seqs', m.get('n_seqs'))" "$WORK/leg_$V/diagnostics.json"
  echo "$(date -Is) LEG_DONE $V" | tee -a "$WORK/timeline.txt"
done
echo "$(date -Is) ALL_DONE" | tee -a "$WORK/timeline.txt"
