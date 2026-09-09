#!/usr/bin/env bash
# E-A' band×SAM3（D092／D094 架構 08-29 定版）：
#   75 支全域三候選觸發（首幀 init；{假色, top-3 band, Fisher 投影}）
#   → 觸發支 band-3ch 轉檔 → SAM3 full band → **crop 窗對 band 軌跡重算**
#   （`hsot.crop_rerun prep`，D094：窗是規則的產物不是規則本身——
#   「同一套規則作用在不同輸入」才是正確單變因；9/7 部署形態的窗本來就重算）
#   → SAM3 crop band → merge → **只對觸發支子集跑 finalize**（後處理全為
#   per-seq/per-frame 操作，子集＝全集的子集結果）→ 拼 sub_v078 成 final75
#   （未觸發支天然位元級＝v078）。不依賴任何 v078 中間產物。
# 組裝雙候選：
#   A（主提交）：source＝既有 sub_v012（SAMURAI 不進變因）——單變因。
#   B（僅歸因）：source＝本次 band-SAMURAI——A/B diff 量「SAMURAI 被偽色影響多少」。
# 沿用既有（獨立腿、非主線導出物）：source=sub_v012、第三腿=rankb_robust_test75
# full_sam3。窗規則同 v078：primary∪source 聯集 envelope、area-frac-max **0.55**
# （profile 實查值；crop_rerun 預設 0.40，必須顯式傳）。
#
# ── SELFKILL：rearm 320 分──────────────────────────────────────────────
# 逐項算術：下載（fc tar/ckpt×2/既有 CSV/首幀 HSI）~15｜decide 75 支 ~5｜
# 觸發支全幀 HSI ~20｜轉檔 ~15｜SAM3 full band ~40｜prep+SAM3 crop ~25｜
# SAMURAI band（full+crop，B 用）~55｜merge+finalize A/B+驗證 ~15｜
# 同機假色對照（SAM3 only）~65｜收尾 ~10 ⇒ ~265 分；rearm 320 留 ~55 分。
# 時間爆了犧牲順序：對照 > B > A（A 完成即先 rclone 落地）。
# 單一計時器慣例同 G1/G2（cloud-init backstop 唯一、不自掛第二顆；08-07 事故）。
set -euo pipefail

export PYTHONUNBUFFERED=1
export PATH="/home/ubuntu/.local/bin:$PATH"
export RCLONE_CONFIG=/home/ubuntu/rclone.conf

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DEST="$GDRIVE/5_outputs/band3ch_sam3_20260829"

die() { echo "FATAL: $*" >&2; exit 1; }

: "${HF_TOKEN:?HF_TOKEN is required (gated facebook/sam3)}"
# 遠端路徑（team-lead 08-29 實查值為預設；fail-fast 保護）：
TEST_HSI_REMOTE="${TEST_HSI_REMOTE:-$GDRIVE/1_data/raw_archive/validation}"  # ranking/ 是空目錄；validation/HSI-{NIR,RedNIR,VIS}/<stem>/*.png 逐幀
SUB_V012_REMOTE="${SUB_V012_REMOTE:-$GDRIVE/5_outputs/submissions/sub_v012_e23b_sam21_ablation.csv}"
SUB_V078_REMOTE="${SUB_V078_REMOTE:-$GDRIVE/5_outputs/submissions/sub_v078_thirdleg_deadzone.csv}"
THIRDLEG_REMOTE="${THIRDLEG_REMOTE:-$GDRIVE/5_outputs/rankb_robust_test75_20260822/run/full_sam3/submission.csv}"

sudo test -x /root/rearm_selfkill.sh \
  || die "缺 /root/rearm_selfkill.sh：cloud-init selfkill backstop 未就位"
sudo /root/rearm_selfkill.sh 320   # 見上方逐項算術

# ⚠️ repo 根不可叫 hsot——與套件 3_src/hsot（namespace package）同名會讓
#    `import hsot` 解析到 repo 根（D093，08-29 演練實測）。
SRC=/home/ubuntu/hsot_repo
WORK=/home/ubuntu/band3ch_sam3
FC=/home/ubuntu/test75_fc
HSI=/home/ubuntu/test75_hsi
BAND=/home/ubuntu/test75_band3ch
CKPT=/home/ubuntu/ckpt
SAM3ENV=/home/ubuntu/sam3env
T1ENV=/home/ubuntu/t1env
SAMURAI=/home/ubuntu/samurai
PY3="$SAM3ENV/bin/python"
PY1="$T1ENV/bin/python"
SAMURAI_SHA=76ba195984892b0d1e3db5d9c9f90bb62175680a
SAM3_SHA=96914d2425f90a64f45ca977c2b5165418099543
SAM3_CKPT_SHA256=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
AREA_FRAC_MAX=0.55   # v078 profile 的 offline_two_pass_crop.area_fraction_max
SYNC_PID=""

test -d "$SRC/3_src" || die "repo 不在 $SRC"

sync_progress() {
  while true; do
    rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 \
      --exclude '*.part' --exclude '*.part/**' || true
    sleep 60
  done
}

finish() {
  RUN_RC=$?
  trap - EXIT
  set +e
  if [ -n "$SYNC_PID" ]; then
    kill "$SYNC_PID" 2>/dev/null
    wait "$SYNC_PID" 2>/dev/null
  fi
  echo "$(date -Is) finish rc=$RUN_RC" | tee -a "$WORK/finish.txt"
  SYNC_RC=0
  if command -v rclone >/dev/null 2>&1 && [ -f "$RCLONE_CONFIG" ]; then
    rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 \
      --exclude '*.part' --exclude '*.part/**' || SYNC_RC=$?
    if [ "$SYNC_RC" -eq 0 ]; then
      rclone check "$WORK" "$DEST" --one-way \
        --exclude '*.part' --exclude '*.part/**' || SYNC_RC=$?
    fi
  else
    SYNC_RC=127
  fi
  if [ "$SYNC_RC" -eq 0 ]; then
    sudo /root/rearm_selfkill.sh 5
  else
    echo "$(date -Is) REMOTE_VERIFY_FAILED rc=$SYNC_RC" | tee -a "$WORK/finish.txt"
    if [ "$RUN_RC" -eq 0 ]; then RUN_RC=3; fi
  fi
  exit "$RUN_RC"
}
trap finish EXIT
trap 'echo "死於第 $LINENO 行" >&2' ERR

mkdir -p "$WORK/logs" "$FC" "$HSI" "$BAND" "$CKPT"
chmod 600 "$RCLONE_CONFIG"
echo "$(date -Is) SETUP_START" | tee "$WORK/timeline.txt"

command -v rclone >/dev/null 2>&1 || { curl -fsSL https://rclone.org/install.sh | sudo bash >/dev/null; }
rclone lsf "$TEST_HSI_REMOTE" >/dev/null || die "TEST_HSI_REMOTE 不可列：$TEST_HSI_REMOTE"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; }

# ── 下載（背景並行）────────────────────────────────────────────────────
(
  set -euo pipefail
  rclone copyto "$GDRIVE/1_data/packed/t1test_fc_75.tar" /home/ubuntu/t1test_fc_75.tar
  tar -xf /home/ubuntu/t1test_fc_75.tar -C "$FC"
  if [ "$(find "$FC" -mindepth 1 -maxdepth 1 -type d | wc -l)" -lt 75 ]; then
    INNER=$(find "$FC" -mindepth 1 -maxdepth 1 -type d | head -1)
    find "$INNER" -mindepth 1 -maxdepth 1 -type d -exec mv {} "$FC"/ \;
    rmdir "$INNER" 2>/dev/null || true
  fi
  test "$(find "$FC" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 75
  echo FC_READY
) > "$WORK/logs/fc_download.log" 2>&1 &
FC_PID=$!

(
  set -euo pipefail
  rclone copyto "$GDRIVE/1_data/raw/sample_submisson.csv" /home/ubuntu/sample_submisson.csv
  rclone copyto "$SUB_V078_REMOTE" /home/ubuntu/sub_v078.csv
  rclone copyto "$SUB_V012_REMOTE" /home/ubuntu/sub_v012.csv
  rclone copyto "$THIRDLEG_REMOTE" /home/ubuntu/thirdleg_full_sam3.csv
  for f in sub_v078 sub_v012 thirdleg_full_sam3; do
    test -s "/home/ubuntu/$f.csv" || { echo "缺 $f.csv"; exit 1; }
  done
  echo BASE_READY
) > "$WORK/logs/base_download.log" 2>&1 &
BASE_PID=$!

(
  set -euo pipefail
  curl -fL -H "Authorization: Bearer $HF_TOKEN" \
    -o "$CKPT/sam3.pt" "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt"
  echo "$SAM3_CKPT_SHA256  $CKPT/sam3.pt" | sha256sum -c -
  rclone copyto "$GDRIVE/4_models/pretrained/sam2.1_hiera_large.pt" "$CKPT/sam2.1_hiera_large.pt"
  test "$(stat -c%s "$CKPT/sam2.1_hiera_large.pt")" -gt 800000000
  echo CKPT_READY
) > "$WORK/logs/ckpt_download.log" 2>&1 &
CKPT_PID=$!

# ── 環境（雙 venv：SAM3 主腿＋SAMURAI band 腿（B 候選歸因用））─────────
[ -x "$T1ENV/bin/python" ] || uv venv --python 3.12 "$T1ENV"
if [ ! -d "$SAMURAI/.git" ]; then
  git clone https://github.com/yangchris11/samurai.git "$SAMURAI"
fi
git -C "$SAMURAI" fetch --depth 1 origin "$SAMURAI_SHA"
git -C "$SAMURAI" checkout --detach "$SAMURAI_SHA"
VIRTUAL_ENV="$T1ENV" uv pip install -q torch torchvision --torch-backend=auto
VIRTUAL_ENV="$T1ENV" uv pip install -q -e "$SAMURAI/sam2" \
  scipy loguru tqdm pandas pillow opencv-python-headless
"$PY1" -c "import torch,sam2,pandas; assert torch.cuda.is_available()"

[ -x "$SAM3ENV/bin/python" ] || uv venv --python 3.12 "$SAM3ENV"
VIRTUAL_ENV="$SAM3ENV" uv pip install -q torch torchvision --torch-backend=auto
VIRTUAL_ENV="$SAM3ENV" uv pip install -q \
  "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}" \
  "numpy<2" ftfy==6.1.1 regex einops psutil scipy av pycocotools numba \
  python-rapidjson pandas pillow tqdm opencv-python-headless pytest
VIRTUAL_ENV="$SAM3ENV" uv pip install -q "setuptools<81"   # D086
"$PY3" -c "import torch,sam3,numpy,pandas; assert torch.cuda.is_available(); assert numpy.__version__.startswith('1.')"

for JOB_PID in "$FC_PID" "$BASE_PID" "$CKPT_PID"; do
  wait "$JOB_PID"
done
grep -q FC_READY "$WORK/logs/fc_download.log"
grep -q BASE_READY "$WORK/logs/base_download.log"
grep -q CKPT_READY "$WORK/logs/ckpt_download.log"

# CPU 單元測試（band 管線 18 個；紅了不燒 GPU）。PYTHONPATH 顯式（D093）。
(cd "$SRC" && PYTHONPATH="$SRC/3_src" "$PY3" -m pytest -q 3_src/test_band3ch_pipeline.py) \
  2>&1 | tee "$WORK/logs/pytest_band.log"

# ── 階段 1：75 支首幀 HSI ＋ 全域三候選觸發判定 ─────────────────────────
echo "$(date -Is) DECIDE_START" | tee -a "$WORK/timeline.txt"
ls "$FC" > "$WORK/seqs75.txt"
test "$(wc -l < "$WORK/seqs75.txt")" -eq 75
# ⚠️ 評估集＝nir+rednir 35 支，VIS 40 支不評估。兩個獨立理由（都要進交付揭露）：
#  (1) 科學面：D092 的 21 支代表性實測，VIS 的光譜相對假色增量為 **−0.001**（2 勝 3 敗）
#      ——官方假色對 VIS 已是好表徵，三候選規則跑下去也會選假色（不觸發），結果相同。
#  (2) 工程面（08-29 實測）：gDrive 上 test 的 VIS 光譜是 **zip 壓縮檔**（`athlete1.zip`），
#      而 NIR/RedNIR 是解開的逐幀 PNG 目錄——型態不同，VIS 需另寫 zip 讀取路徑。
# ⇒ 交付敘述必須誠實寫成「VIS 未評估（前置研究顯示無增益＋資料型態不同）、維持官方輸入」，
#    **不可寫成「規則全開後 VIS 自然落選」**——那是我們沒做的事。
grep -E '^(nir|rednir)-' "$WORK/seqs75.txt" > "$WORK/seqs_eval.txt"
test "$(wc -l < "$WORK/seqs_eval.txt")" -eq 35
echo "評估集 $(wc -l < "$WORK/seqs_eval.txt") 支（nir+rednir）；VIS 40 支維持官方假色" \
  | tee -a "$WORK/timeline.txt"

declare -A MODDIR=( [vis]=HSI-VIS [nir]=HSI-NIR [rednir]=HSI-RedNIR )
while read -r SEQ; do
  MOD="${SEQ%%-*}"
  STEM="${SEQ#*-}"
  RDIR="$TEST_HSI_REMOTE/${MODDIR[$MOD]}/$STEM"
  FIRST=$(rclone lsf "$RDIR" --include "*.png" | sort | head -1)
  test -n "$FIRST" || die "$SEQ: 遠端無 mosaic png（$RDIR）"
  mkdir -p "$HSI/$SEQ"
  rclone copyto "$RDIR/$FIRST" "$HSI/$SEQ/$FIRST"
done < "$WORK/seqs_eval.txt"

PYTHONPATH="$SRC/3_src" "$PY3" "$SRC/3_src/prep/make_band3ch_frames.py" \
  --fc-root "$FC" --hsi-root "$HSI" --seqs "$WORK/seqs_eval.txt" \
  --decide-out "$WORK/triggered.json" \
  2>&1 | tee "$WORK/logs/decide.log"
"$PY3" -c "
import json; p=json.load(open('$WORK/triggered.json'))
assert not p['failed'], f'decide 失敗支：{p[\"failed\"]}'
assert p['n_triggered'] >= 5, f'觸發僅 {p[\"n_triggered\"]} 支——與前測（5 支贏家 5/5 觸發）不符，停工檢查'
open('$WORK/triggered.txt','w').write('\n'.join(p['triggered'])+'\n')
print('triggered', p['n_triggered'], '/', p['n_seqs'])
"

# 子集檔（觸發支）：sample／source(v012)／third-leg——finalize 只對觸發支跑。
"$PY3" - "$WORK" <<'PYEOF'
import csv, sys
from pathlib import Path
work = Path(sys.argv[1])
trig = set(work.joinpath("triggered.txt").read_text().split())
def subset(src, dst):
    rows = list(csv.reader(open(src)))
    keep = [rows[0]] + [r for r in rows[1:] if r and r[0].rsplit("_", 1)[0] in trig]
    csv.writer(open(dst, "w", newline="")).writerows(keep)
    return len(keep) - 1
n1 = subset("/home/ubuntu/sample_submisson.csv", work / "sample_subset.csv")
n2 = subset("/home/ubuntu/sub_v012.csv", work / "source_v012_subset.csv")
n3 = subset("/home/ubuntu/thirdleg_full_sam3.csv", work / "thirdleg_subset.csv")
assert n1 == n2 == n3 and n1 > 0, f"子集行數不一致：sample={n1} v012={n2} thirdleg={n3}"
print("subset rows:", n1)
PYEOF

# ── 階段 2：觸發支全幀 HSI ＋ band-3ch 轉檔 ────────────────────────────
echo "$(date -Is) CONVERT_START" | tee -a "$WORK/timeline.txt"
while read -r SEQ; do
  MOD="${SEQ%%-*}"
  STEM="${SEQ#*-}"
  rclone copy "$TEST_HSI_REMOTE/${MODDIR[$MOD]}/$STEM" "$HSI/$SEQ" \
    --include "*.png" --transfers 16 --checkers 32 &
  while [ "$(jobs -pr | wc -l)" -ge 4 ]; do wait -n; done
done < "$WORK/triggered.txt"
wait

PYTHONPATH="$SRC/3_src" "$PY3" "$SRC/3_src/prep/make_band3ch_frames.py" \
  --fc-root "$FC" --hsi-root "$HSI" --seqs "$WORK/triggered.txt" \
  --triggered-json "$WORK/triggered.json" --out-root "$BAND" \
  2>&1 | tee "$WORK/logs/convert.log"

# ── 階段 3：SAM3 腿（band 主 run 先、fc 對照最後）──────────────────────
sync_progress &
SYNC_PID=$!

run_sam3_leg() {  # $1=tag  $2=frames_root（觸發支輸入）
  local TAG="$1" ROOT="$2"
  echo "$(date -Is) SAM3_FULL_${TAG}_START" | tee -a "$WORK/timeline.txt"
  "$PY3" "$SRC/3_src/track_t1.py" \
    --backend sam3 --sam3-version sam3 --sam3-eval \
    --sam3-ckpt "$CKPT/sam3.pt" \
    --frames-root "$ROOT" \
    --seq-list "$WORK/triggered.txt" \
    --out-dir "$WORK/full_$TAG" \
    --source-revision "$SAM3_SHA" \
    --device cuda:0 \
    2>&1 | tee "$WORK/logs/full_$TAG.log"
  test -s "$WORK/full_$TAG/submission.csv"

  # crop 窗**對本 leg 的 full 軌跡重算**（D094）：規則同 v078——
  # primary∪source(既有 sub_v012) 聯集 envelope、area-frac-max 0.55。
  echo "$(date -Is) CROP_PREP_${TAG}_START" | tee -a "$WORK/timeline.txt"
  PYTHONPATH="$SRC/3_src" "$PY3" -m hsot.crop_rerun prep \
    --frames-root "$ROOT" \
    --base-csv "$WORK/full_$TAG/submission.csv" \
    --envelope-extra /home/ubuntu/sub_v012.csv \
    --out-root "$WORK/crop_frames_$TAG" \
    --meta "$WORK/crop_meta_$TAG.json" \
    --area-frac-max "$AREA_FRAC_MAX" \
    2>&1 | tee "$WORK/logs/crop_prep_$TAG.log"
  "$PY3" -c "
import json
m = json.load(open('$WORK/crop_meta_$TAG.json'))
open('$WORK/crop_names_$TAG.txt','w').write('\n'.join(sorted(m)) + ('\n' if m else ''))
print(f'crop segments: {len(m)}（{len({v[\"seq\"] for v in m.values()})} 支）')
"

  if [ -s "$WORK/crop_names_$TAG.txt" ]; then
    echo "$(date -Is) SAM3_CROP_${TAG}_START" | tee -a "$WORK/timeline.txt"
    "$PY3" "$SRC/3_src/track_t1.py" \
      --backend sam3 --sam3-version sam3 --sam3-eval \
      --sam3-ckpt "$CKPT/sam3.pt" \
      --frames-root "$WORK/crop_frames_$TAG" \
      --seq-list "$WORK/crop_names_$TAG.txt" \
      --out-dir "$WORK/crop_$TAG" \
      --source-revision "$SAM3_SHA" \
      --device cuda:0 \
      2>&1 | tee "$WORK/logs/crop_$TAG.log"
    test -s "$WORK/crop_$TAG/submission.csv"
    PYTHONPATH="$SRC/3_src" "$PY3" -m hsot.crop_rerun merge \
      --base-csv "$WORK/full_$TAG/submission.csv" \
      --crop-csv "$WORK/crop_$TAG/submission.csv" \
      --meta "$WORK/crop_meta_$TAG.json" \
      --out "$WORK/main_${TAG}_subset.csv" \
      2>&1 | tee "$WORK/logs/merge_crop_$TAG.log"
  else
    echo "本 leg 無序列通過 crop 資格——main=full" | tee -a "$WORK/timeline.txt"
    cp "$WORK/full_$TAG/submission.csv" "$WORK/main_${TAG}_subset.csv"
  fi

  # finalize 只對觸發支子集跑（corr/splice/selector/qhead/第三腿全為
  # per-seq/per-frame 操作，子集＝全集的子集結果；selector/qhead 權重是
  # repo 內固化常數）。source＝既有 sub_v012 子集（候選 A 語意）。
  "$PY3" "$SRC/3_src/finalize_submission.py" \
    --main "$WORK/main_${TAG}_subset.csv" \
    --source "$WORK/source_v012_subset.csv" \
    --sample "$WORK/sample_subset.csv" \
    --corr both --K 6 \
    --selector v2 --qhead v056 \
    --third-leg "$WORK/thirdleg_subset.csv" \
    --out "$WORK/final_subset_$TAG.csv" \
    2>&1 | tee "$WORK/logs/finalize_$TAG.log"
  test -s "$WORK/final_subset_$TAG.csv"

  # 拼 sub_v078 成 75 支（未觸發支天然位元級＝v078；merge 內建
  # exact-set／preserve_init／finite 驗證）。
  PYTHONPATH="$SRC/3_src" "$PY3" "$SRC/3_src/prep/merge_band_submission.py" \
    --base-csv /home/ubuntu/sub_v078.csv \
    --band-csv "$WORK/final_subset_$TAG.csv" \
    --triggered-json "$WORK/triggered.json" \
    --sample /home/ubuntu/sample_submisson.csv \
    --fc-root "$FC" \
    --out "$WORK/submission_${TAG}_final75.csv" \
    --report "$WORK/final75_${TAG}_report.json" \
    2>&1 | tee "$WORK/logs/final75_$TAG.log"
  test -s "$WORK/submission_${TAG}_final75.csv"
}

run_sam3_leg band "$BAND"

# sanity：未觸發序列 vs sub_v078 位元級相同（merge 行替換的固有性質，仍驗證）。
"$PY3" - "$WORK" <<'PYEOF'
import csv, json, sys
from pathlib import Path
work = Path(sys.argv[1])
trig = set(json.load(open(work / "triggered.json"))["triggered"])
ours = {r[0]: r for r in list(csv.reader(open(work / "submission_band_final75.csv")))[1:]}
v078 = {r[0]: r for r in list(csv.reader(open("/home/ubuntu/sub_v078.csv")))[1:]}
assert set(ours) == set(v078), "ID 集合與 sub_v078 不同"
diff = [i for i in ours if i.rsplit("_", 1)[0] not in trig and ours[i] != v078[i]]
assert not diff, f"未觸發序列有 {len(diff)} 行與 v078 不同——merge 錯位"
n_changed = sum(1 for i in ours if i.rsplit("_", 1)[0] in trig and ours[i] != v078[i])
print(f"單變因驗證通過：未觸發支位元級=v078；觸發支變動 {n_changed} 行")
PYEOF
echo "$(date -Is) MAIN_RUN_DONE" | tee -a "$WORK/timeline.txt"
rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 \
  --exclude '*.part' --exclude '*.part/**' || true   # 候選 A 先落地 gDrive

# ── 候選 B（僅歸因，不提交）：source＝本次 band-SAMURAI ─────────────────
echo "$(date -Is) SAMURAI_BAND_START" | tee -a "$WORK/timeline.txt"
"$PY1" "$SRC/3_src/track_t1.py" \
  --backend samurai \
  --samurai-dir "$SAMURAI" \
  --ckpt "$CKPT/sam2.1_hiera_large.pt" \
  --frames-root "$BAND" \
  --seq-list "$WORK/triggered.txt" \
  --out-dir "$WORK/samurai_full_band" \
  --source-revision "$SAMURAI_SHA" \
  --device cuda:0 \
  2>&1 | tee "$WORK/logs/samurai_full_band.log"
test -s "$WORK/samurai_full_band/submission.csv"

if [ -s "$WORK/crop_names_band.txt" ]; then
  "$PY1" "$SRC/3_src/track_t1.py" \
    --backend samurai \
    --samurai-dir "$SAMURAI" \
    --ckpt "$CKPT/sam2.1_hiera_large.pt" \
    --frames-root "$WORK/crop_frames_band" \
    --seq-list "$WORK/crop_names_band.txt" \
    --out-dir "$WORK/samurai_crop_band" \
    --source-revision "$SAMURAI_SHA" \
    --device cuda:0 \
    2>&1 | tee "$WORK/logs/samurai_crop_band.log"
  test -s "$WORK/samurai_crop_band/submission.csv"
  PYTHONPATH="$SRC/3_src" "$PY3" -m hsot.crop_rerun merge \
    --base-csv "$WORK/samurai_full_band/submission.csv" \
    --crop-csv "$WORK/samurai_crop_band/submission.csv" \
    --meta "$WORK/crop_meta_band.json" \
    --out "$WORK/source_band_subset.csv" \
    2>&1 | tee "$WORK/logs/merge_crop_samurai.log"
else
  cp "$WORK/samurai_full_band/submission.csv" "$WORK/source_band_subset.csv"
fi

"$PY3" "$SRC/3_src/finalize_submission.py" \
  --main "$WORK/main_band_subset.csv" \
  --source "$WORK/source_band_subset.csv" \
  --sample "$WORK/sample_subset.csv" \
  --corr both --K 6 \
  --selector v2 --qhead v056 \
  --third-leg "$WORK/thirdleg_subset.csv" \
  --out "$WORK/final_subset_B.csv" \
  2>&1 | tee "$WORK/logs/finalize_B.log"
PYTHONPATH="$SRC/3_src" "$PY3" "$SRC/3_src/prep/merge_band_submission.py" \
  --base-csv /home/ubuntu/sub_v078.csv \
  --band-csv "$WORK/final_subset_B.csv" \
  --triggered-json "$WORK/triggered.json" \
  --sample /home/ubuntu/sample_submisson.csv \
  --fc-root "$FC" \
  --out "$WORK/submission_B_bothlegs_final75.csv" \
  --report "$WORK/final75_B_report.json" \
  2>&1 | tee "$WORK/logs/final75_B.log"

# A/B diff：SAMURAI 毒害度的直接讀數（per-seq 報告只算觸發支）
PYTHONPATH="$SRC/3_src" "$PY3" "$SRC/3_src/prep/merge_band_submission.py" \
  --base-csv "$WORK/submission_band_final75.csv" \
  --band-csv "$WORK/submission_B_bothlegs_final75.csv" \
  --triggered-json "$WORK/triggered.json" \
  --sample /home/ubuntu/sample_submisson.csv \
  --fc-root "$FC" \
  --out "$WORK/_diff_ab_discard.csv" \
  --report "$WORK/diff_A_vs_B_samurai_effect.json" \
  2>&1 | tee "$WORK/logs/diff_ab.log" || true
echo "$(date -Is) CANDIDATE_B_DONE" | tee -a "$WORK/timeline.txt"
rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 \
  --exclude '*.part' --exclude '*.part/**' || true

# ── 同機假色對照（歸因用；A/B 已落地，時間爆了損失的只是對照）─────────
run_sam3_leg fc "$FC"
PYTHONPATH="$SRC/3_src" "$PY3" "$SRC/3_src/prep/merge_band_submission.py" \
  --base-csv "$WORK/submission_fc_final75.csv" \
  --band-csv "$WORK/submission_band_final75.csv" \
  --triggered-json "$WORK/triggered.json" \
  --sample /home/ubuntu/sample_submisson.csv \
  --fc-root "$FC" \
  --out "$WORK/_diff_fc_discard.csv" \
  --report "$WORK/diff_band_vs_samemachine_fc.json" \
  2>&1 | tee "$WORK/logs/diff_fc.log" || true

echo "$(date -Is) ALL_DONE" | tee -a "$WORK/timeline.txt"
echo "提交檔（候選 A）：$DEST/submission_band_final75.csv（明早 vs v078=0.71666，判準 ≥+0.001 開通）"
