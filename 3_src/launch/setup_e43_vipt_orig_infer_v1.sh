#!/usr/bin/env bash
# setup_e43_vipt_orig_infer_v1.sh — 作者 ViPT_all.pth.tar 補跑：
#   (1) val VIS orig → local_orig_vis.csv
#   (2) test RedNIR orig → 第三者投票原料
# 在 Lambda 上跑。紀律：D036 子集不整包、D016 階段回傳、D018 結束 terminate。
set -euo pipefail
trap 'echo "🚨 死於第 $LINENO 行 exit=$?"' ERR
export PYTHONUNBUFFERED=1

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
WORK="${WORK:-$HOME/e43}"
REPO="$WORK/ViPT_HOT2023"
DATA="$WORK/data"
OUT="$WORK/out"
LOG="$WORK/e43.log"
VENV="$WORK/venv"
MODE="${1:-FULL}"   # DRY | FULL

mkdir -p "$DATA" "$OUT" "$WORK"
exec > >(tee -a "$LOG") 2>&1
echo "=== E43 $(date -u +%Y-%m-%dT%H:%M:%SZ) MODE=$MODE ==="

if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | sudo bash
fi
rclone lsf "$GDRIVE/" >/dev/null

# --- env（推論；不鎖 1.10，現成 CUDA 映像上裝能跑的 torch）---
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -U pip
  "$VENV/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  "$VENV/bin/pip" install timm opencv-python-headless numpy pillow pyyaml yacs tqdm pandas
fi
PY="$VENV/bin/python"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# --- repo ---
if [ ! -d "$REPO/.git" ]; then
  git clone --depth 1 https://github.com/laisimiao/ViPT_HOT2023.git "$REPO"
fi
# 覆蓋 loader（現場 X2Cube，不吃 npy）
if [ -f "$HOME/hsot_src/t2_vipt/hsi3d.py" ]; then
  cp "$HOME/hsot_src/t2_vipt/hsi3d.py" "$REPO/lib/train/dataset/hsi3d.py"
fi

# --- 權重 ---
rclone copyto "$GDRIVE/4_models/pretrained_t2/ViPT_all.pth.tar" "$WORK/ViPT_all.pth.tar"
ls -lh "$WORK/ViPT_all.pth.tar"

# --- 資料：只拉需要的序列 ---
VAL_SPLIT="${VAL_SPLIT:-$HOME/hsot_src/../1_data/val_split_v1.txt}"
[ -f "$HOME/val_split_v1.txt" ] && VAL_SPLIT="$HOME/val_split_v1.txt"
GT_CSV="${GT_CSV:-$HOME/2026training.csv}"

pull_seq() {
  local src_mod="$1" dest_root="$2" name="$3"
  mkdir -p "$dest_root/$name"
  # training VIS 在 raw_archive 是 zip；validation RedNIR 是目錄。兩種都試。
  if rclone lsf "$GDRIVE/1_data/raw_archive/$src_mod/${name}.zip" >/dev/null 2>&1; then
    local z="/tmp/e43_${name}_$$.zip"
    rclone copyto "$GDRIVE/1_data/raw_archive/$src_mod/${name}.zip" "$z"
    unzip -qo "$z" -d "$dest_root/$name"
    rm -f "$z"
    if [ -d "$dest_root/$name/$name" ]; then
      shopt -s dotglob
      mv "$dest_root/$name/$name"/* "$dest_root/$name/" || true
      rmdir "$dest_root/$name/$name" || true
      shopt -u dotglob
    fi
  else
    rclone copy "$GDRIVE/1_data/raw_archive/$src_mod/$name" "$dest_root/$name" --transfers 8
  fi
}

echo "=== 拉 test RedNIR（validation 子集）==="
mkdir -p "$DATA/test_rn/HSI-RedNIR" "$DATA/test_rn/HSI-RedNIR-FalseColor"
mapfile -t RN_NAMES < <(rclone lsf "$GDRIVE/1_data/raw_archive/validation/HSI-RedNIR-FalseColor" | sed 's:/$::')
echo "RedNIR test 序列 ${#RN_NAMES[@]} 支"
if [ "$MODE" = DRY ]; then
  RN_NAMES=("${RN_NAMES[0]}")
fi
for n in "${RN_NAMES[@]}"; do
  [ -n "$n" ] || continue
  pull_seq "validation/HSI-RedNIR" "$DATA/test_rn/HSI-RedNIR" "$n"
  pull_seq "validation/HSI-RedNIR-FalseColor" "$DATA/test_rn/HSI-RedNIR-FalseColor" "$n"
done
rclone copyto "$GDRIVE/5_outputs/e43_vipt_orig_20260818/.keep" /tmp/.keep 2>/dev/null || true
echo "test RedNIR 就緒 $(find "$DATA/test_rn/HSI-RedNIR-FalseColor" -mindepth 1 -maxdepth 1 -type d | wc -l) 支"

# --- DRY：一支 RedNIR ---
INFER="$HOME/hsot_src/t2_vipt/run_t2_infer.py"
[ -f "$INFER" ] || INFER="$WORK/run_t2_infer.py"
"$PY" "$INFER" --repo "$REPO" --data "$DATA/test_rn" \
  --ckpt "$WORK/ViPT_all.pth.tar" --out "$OUT/dry_rednir.csv" \
  --dataset-mode HOT23TEST --label dry
ls -l "$OUT/dry_rednir.csv"
rclone copy "$OUT/dry_rednir.csv" "$GDRIVE/5_outputs/e43_vipt_orig_20260818/"
echo "=== DRY 過關，已回傳 dry_rednir.csv ==="
[ "$MODE" = DRY ] && exit 0

# --- FULL test RedNIR ---
# 只跑 RedNIR，不能對全量 sample 做 exact-set
"$PY" "$INFER" --repo "$REPO" --data "$DATA/test_rn" \
  --ckpt "$WORK/ViPT_all.pth.tar" --out "$OUT/test_rednir_orig.csv" \
  --dataset-mode HOT23TEST --label testrn
rclone copy "$OUT/test_rednir_orig.csv" "$GDRIVE/5_outputs/e43_vipt_orig_20260818/"
echo "=== test RedNIR orig 已回傳 ==="

# --- val VIS orig ---
echo "=== 拉 val VIS（training 子集）==="
mkdir -p "$DATA/val_vis/HSI-VIS" "$DATA/val_vis/HSI-VIS-FalseColor"
VIS_LIST=()
while read -r s; do
  case "$s" in vis-*) VIS_LIST+=("${s#vis-}") ;; esac
done < "$VAL_SPLIT"
echo "val VIS ${#VIS_LIST[@]} 支"
for n in "${VIS_LIST[@]}"; do
  pull_seq "training/HSI-VIS" "$DATA/val_vis/HSI-VIS" "$n"
  pull_seq "training/HSI-VIS-FalseColor" "$DATA/val_vis/HSI-VIS-FalseColor" "$n"
done
"$PY" - <<PY
from collections import defaultdict
from pathlib import Path
gt_csv = Path("$GT_CSV")
root = Path("$DATA/val_vis")
by = defaultdict(list)
with gt_csv.open() as f:
    next(f)
    for line in f:
        i, x, y, w, h = line.strip().split(",")
        seq, fr = i.rsplit("_", 1)
        if not seq.startswith("vis-"):
            continue
        by[seq].append((int(fr), float(x), float(y), float(w), float(h)))
n_w = 0
for seq, rows in by.items():
    name = seq[4:]
    rows.sort()
    for sub in ("HSI-VIS", "HSI-VIS-FalseColor"):
        d = root / sub / name
        if not d.is_dir():
            continue
        # ViPT genConfig: np.loadtxt(..., delimiter='\t') then fallback default
        # (whitespace). Comma-separated files crash both paths.
        (d / "groundtruth_rect.txt").write_text(
            "\n".join(f"{x:.2f}\t{y:.2f}\t{w:.2f}\t{h:.2f}" for _, x, y, w, h in rows) + "\n"
        )
        n_w += 1
print("wrote groundtruth_rect", n_w)
PY
"$PY" "$INFER" --repo "$REPO" --data "$DATA/val_vis" \
  --ckpt "$WORK/ViPT_all.pth.tar" --out "$OUT/local_orig_vis.csv" \
  --dataset-mode HOT23VAL --label valvis \
  --only-seqs "$VAL_SPLIT"
rclone copy "$OUT/local_orig_vis.csv" "$GDRIVE/5_outputs/e43_vipt_orig_20260818/"
rclone copy "$OUT/local_orig_vis.csv" "$GDRIVE/5_outputs/t2_bench/"
echo "=== val VIS orig 已回傳 ==="
echo "=== E43 FULL 完成 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
