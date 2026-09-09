#!/usr/bin/env bash
# ============================================================================
# e05_e16_night_v1.sh — E05 band selection 前測(8 支 nir 失分序列)
#                      + E16 teacher 偽 mask 生成(65 支假色,stride=5)
# 前置 scp:rclone.conf、.lambda_key、track_t1.py、band_select.py、io_hsot.py、
#          teacher_gen.py、2026training.csv、val_split_v1.txt、e05_seqs.txt
# 跑完全部回傳 gDrive 後自毀(3h 保險兜底)。
# ============================================================================
set -euo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1
GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
P=~/t1env/bin/python

echo "=== [1/6] 環境(samurai 隔離 venv 成熟配方)==="
command -v rclone >/dev/null || (curl -s https://rclone.org/install.sh | sudo bash > /dev/null 2>&1)
[ -d ~/t1env ] || python3 -m venv ~/t1env
$P -m pip install -q --upgrade pip 2>&1 | tail -1
$P -m pip install -q torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -1
[ -d ~/samurai ] || git clone --depth 1 https://github.com/yangchris11/samurai.git ~/samurai
$P -m pip install -q -e ~/samurai/sam2 loguru tqdm pandas pillow scipy 2>&1 | tail -1
$P -c "import sam2, torch; assert torch.cuda.is_available(); print('env OK')"

echo "=== [2/6] 資料 ==="
if [ ! -d ~/t1_data ]; then
  rclone copy "$GDRIVE/1_data/packed/t1val_fc_65.tar" ~/
  mkdir -p ~/t1_data && tar xf ~/t1val_fc_65.tar -C ~/t1_data
fi
echo "假色:$(ls ~/t1_data | wc -l) 支"
mkdir -p ~/ckpt
[ -f ~/ckpt/sam2.1_hiera_large.pt ] || rclone copy "$GDRIVE/4_models/pretrained/sam2.1_hiera_large.pt" ~/ckpt/
mkdir -p ~/hsi_zips ~/hsi_data
while read -r seq; do
  [ -z "$seq" ] && continue
  name="${seq#nir-}"
  dest=~/hsi_data/"$seq"
  if [ -d "$dest" ] && [ "$(find "$dest" -name "*.png" | wc -l)" -gt 0 ]; then continue; fi
  z=~/hsi_zips/"$seq".zip
  if [ ! -f "$z" ]; then
    if ! rclone copyto "$GDRIVE/1_data/raw_archive/training/update/HSI-NIR/$name.zip" "$z" 2>/dev/null; then
      rclone copyto "$GDRIVE/1_data/raw_archive/training/HSI-NIR/$name.zip" "$z"
    fi
  fi
  mkdir -p "$dest"; t=$(mktemp -d)
  unzip -q "$z" -d "$t"
  find "$t" -iname "*.png" -exec mv {} "$dest"/ \;
  rm -rf "$t"
  echo "  HSI $seq: $(find "$dest" -name "*.png" | wc -l) png"
done < ~/e05_seqs.txt

echo "=== [3/6] E05:band-selected 3ch 生成(首幀 init 取自 GT)==="
$P - <<'PYEOF'
import pandas as pd, json
gt = pd.read_csv('/home/ubuntu/2026training.csv')
gt.columns = ['ID','x','y','w','h']
p = gt['ID'].str.rsplit('_', n=1, expand=True)
gt['seq'], gt['frame'] = p[0], p[1].astype(int)
seqs = [s.strip() for s in open('/home/ubuntu/e05_seqs.txt') if s.strip()]
init = {}
for s in seqs:
    r = gt[gt.seq==s].sort_values('frame').iloc[0]
    init[s] = f"{r.x} {r.y} {r.w} {r.h}"
json.dump(init, open('/home/ubuntu/e05_init.json','w'))
print('init OK:', len(init))
PYEOF
mkdir -p ~/band3ch ~/band_reports
while read -r seq; do
  [ -z "$seq" ] && continue
  [ -d ~/band3ch/"$seq" ] && continue
  INIT=$($P -c "import json; print(json.load(open('/home/ubuntu/e05_init.json'))['$seq'])")
  $P ~/band_select.py --mosaic-root ~/hsi_data --seq "$seq" --init "$INIT" \
    --out-dir ~/band3ch --report ~/band_reports/"$seq".json
done < ~/e05_seqs.txt
echo "BAND3CH-DONE"

echo "=== [4/6] E05:T1 跑 band 版(單變因:僅輸入 3ch 不同)==="
$P ~/track_t1.py --frames-root ~/band3ch --gt-csv ~/2026training.csv \
  --out-dir ~/out_e05_band --samurai-dir ~/samurai --ckpt ~/ckpt/sam2.1_hiera_large.pt

echo "=== [5/6] E16 teacher 偽 mask(65 支,stride=5,tightness≥0.8)==="
$P ~/teacher_gen.py --frames-root ~/t1_data --gt-csv ~/2026training.csv \
  --seq-list ~/val_split_v1.txt --samurai-dir ~/samurai --ckpt ~/ckpt/sam2.1_hiera_large.pt \
  --stride 5 --min-tightness 0.8 --out-dir ~/teacher_masks

echo "=== [6/6] 回傳 ==="
rclone copy ~/out_e05_band "$GDRIVE/5_outputs/e05_band_probe" --transfers 8
rclone copy ~/band_reports "$GDRIVE/5_outputs/e05_band_probe/reports" 2>/dev/null || true
rclone copy ~/teacher_masks "$GDRIVE/1_data/teacher_masks_v1" --transfers 8
echo "=== NIGHT CHAIN done ==="

key=$(cat ~/.lambda_key | tr -d '\n')
id=$(curl -s -u "$key:" https://cloud.lambdalabs.com/api/v1/instances | python3 -c "
import json,sys
d=json.load(sys.stdin).get('data',[])
m=[i['id'] for i in d if i.get('name')=='hsot-e05-night']
print(m[0] if m else '')")
[ -n "$id" ] && curl -s -u "$key:" -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
  -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$id\"]}" && echo "TERMINATED"
