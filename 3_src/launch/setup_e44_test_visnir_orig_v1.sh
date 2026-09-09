#!/usr/bin/env bash
# setup_e44_test_visnir_orig_v1.sh — test VIS+NIR 作者 ViPT orig（HOT23TEST）
# 給死區凍結救援當獨立來源。RedNIR 已有，不重跑。
# DRY=一支 VIS；FULL=NIR 再 VIS，每階段 rclone（D016）。
set -euo pipefail
trap 'echo "🚨 死於第 $LINENO 行 exit=$?"; echo FAILED > "${WORK:-$HOME/e44}/status.txt"' ERR
export PYTHONUNBUFFERED=1
export PYTHONPATH="${HOME}/hsot_src/3_src/launch/vot_shim:${HOME}/hsot_src/launch/vot_shim:${PYTHONPATH:-}"

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
WORK="${WORK:-$HOME/e44}"
REPO="$WORK/ViPT_HOT2023"
DATA="$WORK/data"
OUT="$WORK/out"
LOG="$WORK/e44.log"
VENV="$WORK/venv"
MODE="${1:-FULL}"   # DRY | FULL
STATUS="$WORK/status.txt"

mkdir -p "$DATA" "$OUT" "$WORK"
echo RUNNING > "$STATUS"
exec >>"$LOG" 2>&1
echo "=== E44 $(date -u +%Y-%m-%dT%H:%M:%SZ) MODE=$MODE ==="

if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | sudo bash
fi
rclone lsf "$GDRIVE/" >/dev/null

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -U pip
  "$VENV/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  "$VENV/bin/pip" install timm opencv-python-headless numpy pillow pyyaml yacs tqdm pandas \
    tensorboardX easydict lmdb pycocotools matplotlib
  "$VENV/bin/pip" install jpeg4py || true
fi
PY="$VENV/bin/python"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [ ! -d "$REPO/.git" ]; then
  git clone --depth 1 https://github.com/laisimiao/ViPT_HOT2023.git "$REPO"
fi
if [ -f "$HOME/hsot_src/t2_vipt/hsi3d.py" ]; then
  cp "$HOME/hsot_src/t2_vipt/hsi3d.py" "$REPO/lib/train/dataset/hsi3d.py"
elif [ -f "$HOME/hsot_src/3_src/t2_vipt/hsi3d.py" ]; then
  cp "$HOME/hsot_src/3_src/t2_vipt/hsi3d.py" "$REPO/lib/train/dataset/hsi3d.py"
fi

# torch._six shim（新 torch 已移除）
LOADER="$REPO/lib/train/data/loader.py"
if grep -q "from torch._six import" "$LOADER"; then
  sed -i 's/from torch._six import string_classes/string_classes = (str, bytes)/' "$LOADER"
fi

# visdom stub（pip 在新 setuptools 會炸）
if ! "$PY" -c "import visdom" 2>/dev/null; then
  mkdir -p "$VENV/lib/python3.10/site-packages/visdom"
  cat > "$VENV/lib/python3.10/site-packages/visdom/__init__.py" <<'PY'
class Visdom:
    def __init__(self, *a, **k): pass
    def __getattr__(self, name):
        def _(*a, **k): return None
        return _
PY
  : > "$VENV/lib/python3.10/site-packages/visdom/server.py"
fi

# local.py
LOCAL="$REPO/lib/test/evaluation/local.py"
if [ ! -f "$LOCAL" ]; then
  cat > "$LOCAL" <<PY
from lib.test.evaluation.environment import EnvSettings
def local_env_settings():
    s = EnvSettings()
    s.prj_dir = "$REPO"
    s.save_dir = "$REPO/output"
    s.results_path = "$REPO/results"
    return s
PY
fi

rclone copyto "$GDRIVE/4_models/pretrained_t2/ViPT_all.pth.tar" "$WORK/ViPT_all.pth.tar"
ls -lh "$WORK/ViPT_all.pth.tar"

pull_seq() {
  local src_mod="$1" dest_root="$2" name="$3"
  mkdir -p "$dest_root/$name"
  if rclone lsf "$GDRIVE/1_data/raw_archive/$src_mod/${name}.zip" >/dev/null 2>&1; then
    local z="/tmp/e44_${name}_$$.zip"
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

INFER="$HOME/hsot_src/t2_vipt/run_t2_infer.py"
[ -f "$INFER" ] || INFER="$HOME/hsot_src/3_src/t2_vipt/run_t2_infer.py"
[ -f "$INFER" ] || INFER="$WORK/run_t2_infer.py"

# --- DRY: 一支 VIS ---
echo "=== DRY 一支 VIS ==="
mkdir -p "$DATA/dry/HSI-VIS" "$DATA/dry/HSI-VIS-FalseColor"
DRY_NAME=$(rclone lsf "$GDRIVE/1_data/raw_archive/validation/HSI-VIS-FalseColor" | sed 's:/$::' | head -1)
echo "DRY seq $DRY_NAME"
pull_seq "validation/HSI-VIS" "$DATA/dry/HSI-VIS" "$DRY_NAME"
pull_seq "validation/HSI-VIS-FalseColor" "$DATA/dry/HSI-VIS-FalseColor" "$DRY_NAME"
ls "$DATA/dry/HSI-VIS-FalseColor/$DRY_NAME" | head
test -f "$DATA/dry/HSI-VIS-FalseColor/$DRY_NAME/init_rect.txt" || \
  test -f "$DATA/dry/HSI-VIS/$DRY_NAME/init_rect.txt" || \
  echo "WARN no init_rect in DRY"
"$PY" "$INFER" --repo "$REPO" --data "$DATA/dry" \
  --ckpt "$WORK/ViPT_all.pth.tar" --out "$OUT/dry_vis.csv" \
  --dataset-mode HOT23TEST --label dryvis
ls -l "$OUT/dry_vis.csv"
rclone copy "$OUT/dry_vis.csv" "$GDRIVE/5_outputs/e44_test_visnir_orig_20260819/"
echo "=== DRY 過關 ==="
[ "$MODE" = DRY ] && { echo DRY_DONE > "$STATUS"; exit 0; }

# --- FULL NIR ---
echo "=== 拉 test NIR ==="
mkdir -p "$DATA/test/HSI-NIR" "$DATA/test/HSI-NIR-FalseColor"
mapfile -t NIR_NAMES < <(rclone lsf "$GDRIVE/1_data/raw_archive/validation/HSI-NIR-FalseColor" | sed 's:/$::')
echo "NIR ${#NIR_NAMES[@]} 支"
for n in "${NIR_NAMES[@]}"; do
  [ -n "$n" ] || continue
  echo "pull nir $n"
  pull_seq "validation/HSI-NIR" "$DATA/test/HSI-NIR" "$n"
  pull_seq "validation/HSI-NIR-FalseColor" "$DATA/test/HSI-NIR-FalseColor" "$n"
done
"$PY" "$INFER" --repo "$REPO" --data "$DATA/test" \
  --ckpt "$WORK/ViPT_all.pth.tar" --out "$OUT/test_nir_orig.csv" \
  --dataset-mode HOT23TEST --label testnir
rclone copy "$OUT/test_nir_orig.csv" "$GDRIVE/5_outputs/e44_test_visnir_orig_20260819/"
rclone copy "$OUT/test_nir_orig.csv" "$GDRIVE/5_outputs/t2_bench/"
echo "=== test NIR orig 已回傳 $(wc -l < "$OUT/test_nir_orig.csv") 列 ==="

# --- FULL VIS（加進同一 data 根，但只跑 VIS 資料夾：清掉 NIR 以免重跑）---
echo "=== 拉 test VIS ==="
mkdir -p "$DATA/testv/HSI-VIS" "$DATA/testv/HSI-VIS-FalseColor"
mapfile -t VIS_NAMES < <(rclone lsf "$GDRIVE/1_data/raw_archive/validation/HSI-VIS-FalseColor" | sed 's:/$::')
echo "VIS ${#VIS_NAMES[@]} 支"
for n in "${VIS_NAMES[@]}"; do
  [ -n "$n" ] || continue
  echo "pull vis $n"
  pull_seq "validation/HSI-VIS" "$DATA/testv/HSI-VIS" "$n"
  pull_seq "validation/HSI-VIS-FalseColor" "$DATA/testv/HSI-VIS-FalseColor" "$n"
done
"$PY" "$INFER" --repo "$REPO" --data "$DATA/testv" \
  --ckpt "$WORK/ViPT_all.pth.tar" --out "$OUT/test_vis_orig.csv" \
  --dataset-mode HOT23TEST --label testvis
rclone copy "$OUT/test_vis_orig.csv" "$GDRIVE/5_outputs/e44_test_visnir_orig_20260819/"
rclone copy "$OUT/test_vis_orig.csv" "$GDRIVE/5_outputs/t2_bench/"
echo "=== test VIS orig 已回傳 $(wc -l < "$OUT/test_vis_orig.csv") 列 ==="

echo "=== E44 FULL 完成 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo DONE > "$STATUS"
