#!/usr/bin/env bash
# ============================================================================
# setup_e46_sam3_crop_scores_v1.sh — SAM3 + crop-zoom(0.55) 同一輪寫 obj_score
#
# 為何：E45 是無 crop 的 SAM3，量到的 score 閘不能貼到 v056（A＝crop-SAM3）。
# 本輪複製 Ranking B 的 A 構造（D073）：
#   窗只算一次＝歷史 e15 ∪ e02、面積門檻 0.55
#   full SAM3 → crop SAM3 → merge 框＋分數（裁切序列用 crop 輪的 score）
# VAL 另把全域幀號 rebase 成 1-based 再 prep（crop_rerun.jpgs[fr-1] 的契約）。
#
# 用法：INSTANCE_ID=... bash setup_e46_sam3_crop_scores_v1.sh [VAL|TEST|BOTH]
# 預設 BOTH。每階段 rclone（D016）。結束 terminate $INSTANCE_ID（D018）。
# SAM3 SHA 釘 96914d24…（D059）。不掃帳號、不 terminate 別台。
# ============================================================================
set -euo pipefail
trap 'echo "🚨 死於第 $LINENO 行 exit=$?"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

INSTANCE_ID="${INSTANCE_ID:?必須傳入 INSTANCE_ID}"
MODE="${1:-BOTH}"
GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DEST="$GDRIVE/5_outputs/e46_sam3_crop_scores_20260819"
WORK="${WORK:-$HOME/e46}"
PY="$WORK/sam3env/bin/python"
SAM3_SHA=96914d2425f90a64f45ca977c2b5165418099543
mkdir -p "$WORK" "$WORK/ckpt" "$WORK/out"
STAMP="$WORK/timing.txt"; : > "$STAMP"
STATUS="$WORK/status.txt"
echo RUNNING > "$STATUS"
exec >>"$WORK/e46.log" 2>&1

mark() { echo "$1 $(date +%s)" >> "$STAMP"; echo "⏱  $1 @ $(date -u +%H:%M:%S)"; }
selfkill() {
  key=$(tr -d '\n' < ~/.lambda_key 2>/dev/null || tr -d '\n' < /root/.lambda_key)
  [ -z "$key" ] && return 0
  curl -s -u "$key:" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H 'Content-Type: application/json' -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}"
}
die() {
  echo "🚨 $*"
  echo "DIED: $*" > "$WORK/DIED.txt"
  echo FAILED > "$STATUS"
  if command -v rclone >/dev/null 2>&1; then
    rclone copy "$WORK/out" "$DEST" --transfers 8 2>/dev/null || true
    rclone copyto "$WORK/e46.log" "$DEST/e46.log" 2>/dev/null || true
    rclone copyto "$STAMP" "$DEST/timing.txt" 2>/dev/null || true
    rclone copyto "$WORK/DIED.txt" "$DEST/DIED.txt" 2>/dev/null || true
  fi
  selfkill
  exit 1
}

mark E46_START
echo "=== E46 $(date -u +%Y-%m-%dT%H:%M:%SZ) MODE=$MODE id=$INSTANCE_ID ==="

# --- rclone ---
if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | sudo bash || die "rclone 安裝失敗"
fi
rclone lsf "$GDRIVE/1_data/" --max-depth 1 >/dev/null || die "rclone 讀不到 gDrive"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

T1="$HOME/hsot_src/3_src/track_t1.py"
[ -f "$T1" ] || T1="$HOME/track_t1.py"
[ -f "$T1" ] || die "track_t1.py 缺"
cp "$T1" "$WORK/track_t1.py"
mkdir -p "$HOME/hsot" "$WORK/hsot"
if [ -d "$HOME/hsot" ] && [ -f "$HOME/hsot/crop_rerun.py" ]; then
  cp -a "$HOME/hsot/"*.py "$WORK/hsot/" 2>/dev/null || true
fi
touch "$WORK/hsot/__init__.py" "$HOME/hsot/__init__.py"
# PYTHONPATH：crop_rerun 在 ~/hsot 或 $WORK/hsot
export PYTHONPATH="$WORK:$HOME:${PYTHONPATH:-}"

# --- 平行：資料 + 權重 + 環境 ---
(
  set -euo pipefail
  mkdir -p "$WORK/val_fc" "$WORK/test_fc"
  if [ "$MODE" != TEST ]; then
    rclone copyto "$GDRIVE/1_data/packed/t1val_fc_65.tar" "$WORK/t1val.tar"
    tar -xf "$WORK/t1val.tar" -C "$WORK/val_fc"
    if [ ! -d "$WORK/val_fc/vis-ant" ] && [ -d "$WORK/val_fc/t1val_fc_65" ]; then
      mv "$WORK/val_fc/t1val_fc_65"/* "$WORK/val_fc/" || true
    fi
    # 再扁一層
    if [ ! -d "$WORK/val_fc/vis-ant" ]; then
      d=$(find "$WORK/val_fc" -mindepth 2 -maxdepth 2 -type d -name 'vis-ant' -print -quit || true)
      [ -n "$d" ] && mv "$(dirname "$d")"/* "$WORK/val_fc/" || true
    fi
    rclone copyto "$GDRIVE/1_data/val_split_v1.txt" "$WORK/val_split_v1.txt"
    rclone copyto "$GDRIVE/1_data/raw/2026training.csv" "$WORK/2026training.csv"
  fi
  if [ "$MODE" != VAL ]; then
    rclone copyto "$GDRIVE/1_data/packed/t1test_fc_75.tar" "$WORK/t1test.tar"
    tar -xf "$WORK/t1test.tar" -C "$WORK/test_fc"
    if [ ! -d "$WORK/test_fc/nir-bee2" ]; then
      d=$(find "$WORK/test_fc" -mindepth 1 -maxdepth 3 -type d -name 'nir-bee2' -print -quit || true)
      [ -n "$d" ] && { src=$(dirname "$d"); find "$src" -mindepth 1 -maxdepth 1 -type d -exec mv {} "$WORK/test_fc/" \; ; }
    fi
    rclone copyto "$GDRIVE/5_outputs/submissions/sub_v006_e15sam3.csv" "$WORK/e15_test.csv"
    rclone copyto "$GDRIVE/5_outputs/submissions/exp003_samurai_large.csv" "$WORK/e02_test.csv"
  fi
  echo DATA_READY
) > "$WORK/data.log" 2>&1 &
DATA_PID=$!

(
  set -euo pipefail
  [ -f "$WORK/ckpt/sam3.pt" ] || rclone copyto "$GDRIVE/4_models/pretrained/sam3.pt" "$WORK/ckpt/sam3.pt" \
    || curl -fL -o "$WORK/ckpt/sam3.pt" "https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt"
  sz=$(stat -c%s "$WORK/ckpt/sam3.pt")
  [ "$sz" -gt 3000000000 ] || exit 1
  echo CKPT_READY
) > "$WORK/ckpt.log" 2>&1 &
CKPT_PID=$!

[ -d "$WORK/sam3env" ] || uv venv --python 3.12 "$WORK/sam3env"
if [ ! -f "$WORK/sam3env/.deps_done" ]; then
  VIRTUAL_ENV="$WORK/sam3env" uv pip install -q torch torchvision --torch-backend=auto
  VIRTUAL_ENV="$WORK/sam3env" uv pip install -q "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}" \
    "numpy<2" ftfy==6.1.1 regex einops psutil scipy av \
    pycocotools numba python-rapidjson pandas pillow tqdm opencv-python-headless
  VIRTUAL_ENV="$WORK/sam3env" uv pip install -q "setuptools<81"
  touch "$WORK/sam3env/.deps_done"
fi
"$PY" -c "import torch,sam3,numpy,pandas; assert torch.cuda.is_available(); assert numpy.__version__.startswith('1.'); print('sam3env OK', torch.__version__)" \
  2>&1 | grep -viE 'warning|deprecat|^ +import' || die "sam3env 驗證失敗"

wait $DATA_PID || true
wait $CKPT_PID || true
grep -q DATA_READY "$WORK/data.log" || die "資料失敗：$(tail -20 "$WORK/data.log")"
grep -q CKPT_READY "$WORK/ckpt.log" || die "權重失敗：$(tail -10 "$WORK/ckpt.log")"
mark DATA_ENV_READY
rclone copyto "$WORK/e46.log" "$DEST/e46.log" || true

# 信封 CSV：本機 deploy 到 $HOME/e46_envelopes/ 或 gDrive
ENVDIR="$HOME/e46_envelopes"
mkdir -p "$ENVDIR"
if [ "$MODE" != TEST ]; then
  [ -f "$ENVDIR/e15_val.csv" ] || rclone copyto "$GDRIVE/5_outputs/e15_sam3_20260806/submission_val65.csv" "$ENVDIR/e15_val.csv" || true
  [ -f "$ENVDIR/e02_val.csv" ] || rclone copyto "$GDRIVE/5_outputs/e02_kfreset_20260806/submission_val65.csv" "$ENVDIR/e02_val.csv" || true
  [ -f "$ENVDIR/e15_val.csv" ] && [ -f "$ENVDIR/e02_val.csv" ] || die "缺 val envelope CSV"
fi

rebase_1based() {
  "$PY" - "$1" "$2" <<'PY'
import csv, sys
from collections import defaultdict
src, dst = sys.argv[1], sys.argv[2]
rows=list(csv.DictReader(open(src)))
by=defaultdict(list)
for r in rows:
    s, fr = r["ID"].rsplit("_", 1)
    by[s].append((int(fr), r))
out=[]
for s, items in by.items():
    items.sort(key=lambda t: t[0])
    for i, (_, r) in enumerate(items, 1):
        out.append({"ID": f"{s}_{i}", "x": r["x"], "y": r["y"],
                    "width": r["width"], "height": r["height"]})
# 穩定順序：原檔出現的序列順序
order=[]
seen=set()
for r in rows:
    s=r["ID"].rsplit("_",1)[0]
    if s not in seen:
        seen.add(s); order.append(s)
out_sorted=[]
for s in order:
    out_sorted.extend([r for r in out if r["ID"].rsplit("_",1)[0]==s])
with open(dst,"w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["ID","x","y","width","height"])
    w.writeheader(); w.writerows(out_sorted)
print(f"rebase {src} → {dst}  {len(out_sorted)} 列")
PY
}

map_1based_to_global() {
  # $1 global template csv  $2 one-based csv  $3 one-based scores or NONE  $4 out csv  $5 out scores or NONE
  "$PY" - "$1" "$2" "$3" "$4" "$5" <<'PY'
import csv, sys
from collections import defaultdict
gpath, opath, spath, outb, outs = sys.argv[1:6]
def load(p):
    rows=list(csv.DictReader(open(p)))
    by=defaultdict(list)
    order=[]
    seen=set()
    for r in rows:
        s=r["ID"].rsplit("_",1)[0]
        if s not in seen:
            seen.add(s); order.append(s)
        by[s].append(r)
    for s in by:
        by[s].sort(key=lambda r: int(r["ID"].rsplit("_",1)[1]))
    return order, by
gord, gby = load(gpath)
oord, oby = load(opath)
assert gord==oord, f"seq order mismatch {len(gord)} vs {len(oord)}"
sc=None
if spath not in ("NONE","", "none"):
    sc={}
    for r in csv.DictReader(open(spath)):
        sc[r["ID"]]=r.get("obj_score","")
bout=[]; sout=[]
for s in gord:
    g, o = gby[s], oby[s]
    assert len(g)==len(o), f"{s}: {len(g)} vs {len(o)}"
    for gr, orow in zip(g, o):
        bout.append({"ID": gr["ID"], "x": orow["x"], "y": orow["y"],
                     "width": orow["width"], "height": orow["height"]})
        if sc is not None:
            sout.append({"ID": gr["ID"], "obj_score": sc.get(orow["ID"], "")})
with open(outb,"w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["ID","x","y","width","height"]); w.writeheader(); w.writerows(bout)
if sc is not None:
    with open(outs,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["ID","obj_score"]); w.writeheader(); w.writerows(sout)
print(f"map → {outb} {len(bout)} 列")
PY
}

merge_scores() {
  "$PY" - "$1" "$2" "$3" "$4" <<'PY'
import csv, json, sys
from collections import defaultdict
base_p, crop_p, meta_p, out_p = sys.argv[1:5]
base={r["ID"]: r.get("obj_score","") for r in csv.DictReader(open(base_p))}
order=[r["ID"] for r in csv.DictReader(open(base_p))]
crop_rows=list(csv.DictReader(open(crop_p)))
by=defaultdict(list)
for r in crop_rows:
    s, fr = r["ID"].rsplit("_", 1)
    by[s].append((int(fr), r.get("obj_score","")))
meta=json.load(open(meta_p))
n=0
for name, w in meta.items():
    items=sorted(by.get(name, []))
    if not items:
        continue
    seq=w.get("seq", name)
    f_lo=int(w.get("frames", [1, None])[0])
    for fr, sc in items:
        gid=f"{seq}_{f_lo + int(fr) - 1}"
        if gid in base:
            base[gid]=sc
            n+=1
with open(out_p,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["ID","obj_score"])
    for i in order:
        w.writerow([i, base[i]])
print(f"scores merge 覆蓋 {n} 列 → {out_p}")
PY
}

run_sam3() {
  local frames="$1" list="$2" outd="$3"
  shift 3
  mkdir -p "$outd"
  "$PY" "$WORK/track_t1.py" --frames-root "$frames" --seq-list "$list" \
    --out-dir "$outd" --backend sam3 --sam3-ckpt "$WORK/ckpt/sam3.pt" "$@"
  test -f "$outd/submission.csv" || die "缺 $outd/submission.csv"
  test -f "$outd/obj_scores.csv" || die "缺 $outd/obj_scores.csv"
}

crop_pass() {
  local frames="$1" e15="$2" e02="$3" tag="$4"
  local crop="$WORK/crop_$tag" meta="$WORK/meta_$tag.json" list="$WORK/crop_seqs_$tag.txt"
  echo "=== crop prep $tag 窗=e15∪e02 門檻 0.55 ==="
  PYTHONPATH="$WORK:$HOME" "$PY" -m hsot.crop_rerun prep --frames-root "$frames" \
    --base-csv "$e15" --envelope-extra "$e02" --area-frac-max 0.55 \
    --out-root "$crop" --meta "$meta" | tail -20
  "$PY" -c "
import json,os
m=json.load(open('$meta'))
open('$list','w').write('\\n'.join(sorted(m))+'\\n')
print('crop 選中', len(m), '支')
"
  local n; n=$(grep -c . "$list" || echo 0)
  [ "$n" -ge 1 ] || die "crop $tag 選中 0 支"
  echo "=== crop SAM3 $tag ($n 支) ==="
  run_sam3 "$crop" "$list" "$WORK/out/crop_$tag"
  PYTHONPATH="$WORK:$HOME" "$PY" -m hsot.crop_rerun merge \
    --base-csv "$WORK/out/${tag}_full/submission.csv" \
    --crop-csv "$WORK/out/crop_$tag/submission.csv" \
    --meta "$meta" --out "$WORK/out/${tag}_merged.csv"
  merge_scores "$WORK/out/${tag}_full/obj_scores.csv" \
    "$WORK/out/crop_$tag/obj_scores.csv" "$meta" \
    "$WORK/out/${tag}_merged_scores.csv"
  rclone copyto "$meta" "$DEST/${tag}_meta.json"
  rclone copyto "$WORK/out/${tag}_merged.csv" "$DEST/${tag}_merged.csv"
  rclone copyto "$WORK/out/${tag}_merged_scores.csv" "$DEST/${tag}_merged_scores.csv"
  rclone copy "$WORK/out/${tag}_full" "$DEST/${tag}_full" --transfers 4 || true
  rclone copy "$WORK/out/crop_$tag" "$DEST/crop_$tag" --transfers 4 || true
}

# ---------- DRY：val 一支，確認 scores 寫得出來 ----------
if [ "$MODE" != TEST ]; then
  VAL_ROOT=$(find "$WORK/val_fc" -mindepth 1 -maxdepth 1 -type d -name 'vis-ant' -print -quit | xargs dirname)
  [ -d "$VAL_ROOT/vis-ant" ] || die "val 假色根目錄找不到 vis-ant"
  echo "val frames-root=$VAL_ROOT n=$(find "$VAL_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  DRY_LIST="$WORK/dry.txt"
  grep -m1 '^rednir-' "$WORK/val_split_v1.txt" > "$DRY_LIST" || grep -m1 '^vis-' "$WORK/val_split_v1.txt" > "$DRY_LIST"
  echo "DRY $(cat "$DRY_LIST")"
  run_sam3 "$VAL_ROOT" "$DRY_LIST" "$WORK/out/dry" --gt-csv "$WORK/2026training.csv"
  rclone copy "$WORK/out/dry" "$DEST/dry" --transfers 4
  echo "=== DRY 過關 ==="
  mark DRY_DONE
fi

# ---------- VAL ----------
if [ "$MODE" != TEST ]; then
  echo "=== VAL full SAM3 65 支 ==="
  run_sam3 "$VAL_ROOT" "$WORK/val_split_v1.txt" "$WORK/out/val_full" --gt-csv "$WORK/2026training.csv"
  mark VAL_FULL_DONE
  rclone copy "$WORK/out/val_full/submission.csv" "$DEST/val_full/" --transfers 4
  rclone copy "$WORK/out/val_full/obj_scores.csv" "$DEST/val_full/" --transfers 4
  rclone copy "$WORK/out/val_full/diagnostics.json" "$DEST/val_full/" --transfers 4

  rebase_1based "$ENVDIR/e15_val.csv" "$WORK/e15_val_1b.csv"
  rebase_1based "$ENVDIR/e02_val.csv" "$WORK/e02_val_1b.csv"
  rebase_1based "$WORK/out/val_full/submission.csv" "$WORK/out/val_full/submission_1b.csv"
  # scores rebase：ID 對齊 submission_1b
  "$PY" - "$WORK/out/val_full/obj_scores.csv" "$WORK/out/val_full/submission.csv" "$WORK/out/val_full/obj_scores_1b.csv" <<'PY'
import csv, sys
from collections import defaultdict
sc={r["ID"]: r["obj_score"] for r in csv.DictReader(open(sys.argv[1]))}
rows=list(csv.DictReader(open(sys.argv[2])))
by=defaultdict(list)
order=[]; seen=set()
for r in rows:
    s=r["ID"].rsplit("_",1)[0]
    if s not in seen:
        seen.add(s); order.append(s)
    by[s].append(r["ID"])
out=[]
for s in order:
    ids=sorted(by[s], key=lambda i: int(i.rsplit("_",1)[1]))
    for i, gid in enumerate(ids, 1):
        out.append({"ID": f"{s}_{i}", "obj_score": sc.get(gid,"")})
with open(sys.argv[3],"w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["ID","obj_score"]); w.writeheader(); w.writerows(out)
print("scores 1b", len(out))
PY

  echo "=== VAL crop（信封＝歷史 e15∪e02 的 1-based）==="
  PYTHONPATH="$WORK:$HOME" "$PY" -m hsot.crop_rerun prep --frames-root "$VAL_ROOT" \
    --base-csv "$WORK/e15_val_1b.csv" --envelope-extra "$WORK/e02_val_1b.csv" \
    --area-frac-max 0.55 --out-root "$WORK/crop_val" --meta "$WORK/meta_val.json" | tail -20
  "$PY" -c "
import json
m=json.load(open('$WORK/meta_val.json'))
open('$WORK/crop_seqs_val.txt','w').write('\\n'.join(sorted(m))+'\\n')
print('val crop 選中', len(m))
"
  [ "$(grep -c . "$WORK/crop_seqs_val.txt")" -ge 1 ] || die "val crop 0 支"
  run_sam3 "$WORK/crop_val" "$WORK/crop_seqs_val.txt" "$WORK/out/crop_val"
  # merge 到 1-based full，再映回全域
  PYTHONPATH="$WORK:$HOME" "$PY" -m hsot.crop_rerun merge \
    --base-csv "$WORK/out/val_full/submission_1b.csv" \
    --crop-csv "$WORK/out/crop_val/submission.csv" \
    --meta "$WORK/meta_val.json" --out "$WORK/out/val_merged_1b.csv"
  merge_scores "$WORK/out/val_full/obj_scores_1b.csv" \
    "$WORK/out/crop_val/obj_scores.csv" "$WORK/meta_val.json" \
    "$WORK/out/val_merged_scores_1b.csv"
  map_1based_to_global "$WORK/out/val_full/submission.csv" \
    "$WORK/out/val_merged_1b.csv" "$WORK/out/val_merged_scores_1b.csv" \
    "$WORK/out/val_merged.csv" "$WORK/out/val_merged_scores.csv"
  rclone copyto "$WORK/meta_val.json" "$DEST/val_meta.json"
  rclone copyto "$WORK/out/val_merged.csv" "$DEST/val_merged.csv"
  rclone copyto "$WORK/out/val_merged_scores.csv" "$DEST/val_merged_scores.csv"
  rclone copy "$WORK/out/crop_val" "$DEST/crop_val" --transfers 4 || true
  mark VAL_DONE
  echo "=== VAL 完成 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
fi

# ---------- TEST ----------
if [ "$MODE" != VAL ]; then
  TEST_ROOT=$(find "$WORK/test_fc" -mindepth 1 -maxdepth 1 -type d -name 'nir-bee2' -print -quit | xargs dirname)
  [ -d "$TEST_ROOT/nir-bee2" ] || die "test 假色找不到 nir-bee2"
  find "$TEST_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort > "$WORK/test75.txt"
  ntest=$(grep -c . "$WORK/test75.txt")
  echo "test frames-root=$TEST_ROOT n=$ntest"
  [ "$ntest" -eq 75 ] || die "test 序列數 $ntest ≠ 75"
  echo "=== TEST full SAM3 75 支（不傳 --gt-csv）==="
  run_sam3 "$TEST_ROOT" "$WORK/test75.txt" "$WORK/out/test_full"
  mark TEST_FULL_DONE
  rclone copy "$WORK/out/test_full/submission.csv" "$DEST/test_full/" --transfers 4
  rclone copy "$WORK/out/test_full/obj_scores.csv" "$DEST/test_full/" --transfers 4
  rclone copy "$WORK/out/test_full/diagnostics.json" "$DEST/test_full/" --transfers 4

  echo "=== TEST crop 窗＝歷史 v006 ∪ exp003 門檻 0.55 ==="
  PYTHONPATH="$WORK:$HOME" "$PY" -m hsot.crop_rerun prep --frames-root "$TEST_ROOT" \
    --base-csv "$WORK/e15_test.csv" --envelope-extra "$WORK/e02_test.csv" \
    --area-frac-max 0.55 --out-root "$WORK/crop_test" --meta "$WORK/meta_test.json" | tail -20
  "$PY" -c "
import json
m=json.load(open('$WORK/meta_test.json'))
open('$WORK/crop_seqs_test.txt','w').write('\\n'.join(sorted(m))+'\\n')
print('test crop 選中', len(m))
"
  [ "$(grep -c . "$WORK/crop_seqs_test.txt")" -ge 1 ] || die "test crop 0 支"
  run_sam3 "$WORK/crop_test" "$WORK/crop_seqs_test.txt" "$WORK/out/crop_test"
  PYTHONPATH="$WORK:$HOME" "$PY" -m hsot.crop_rerun merge \
    --base-csv "$WORK/out/test_full/submission.csv" \
    --crop-csv "$WORK/out/crop_test/submission.csv" \
    --meta "$WORK/meta_test.json" --out "$WORK/out/test_merged.csv"
  merge_scores "$WORK/out/test_full/obj_scores.csv" \
    "$WORK/out/crop_test/obj_scores.csv" "$WORK/meta_test.json" \
    "$WORK/out/test_merged_scores.csv"
  rclone copyto "$WORK/meta_test.json" "$DEST/test_meta.json"
  rclone copyto "$WORK/out/test_merged.csv" "$DEST/test_merged.csv"
  rclone copyto "$WORK/out/test_merged_scores.csv" "$DEST/test_merged_scores.csv"
  rclone copy "$WORK/out/crop_test" "$DEST/crop_test" --transfers 4 || true
  mark TEST_DONE
  echo "=== TEST 完成 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
fi

rclone copyto "$WORK/e46.log" "$DEST/e46.log" || true
rclone copyto "$STAMP" "$DEST/timing.txt" || true
echo DONE > "$STATUS"
rclone copyto "$STATUS" "$DEST/status.txt" || true
echo "✅ E46 完成 MODE=$MODE"
echo "🔻 terminate $INSTANCE_ID"
selfkill
