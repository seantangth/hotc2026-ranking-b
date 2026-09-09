#!/usr/bin/env bash
# ============================================================================
# setup_e25b_v2.sh — 接力（同機）：gate v2 on 腿重跑 ＋ B 線 mosaic 三支擴測
#
# 前置：setup_e25_gate_canary_v1.sh 已完成（off 腿在 ~/e25_off 可復用、
#       sam3env/權重/val 假色/GT 已就位、機器 4h 保險自毀運行中）
#
# [A-v2] gate v2 = v1 + 持續運動判別（cont_frac=0.5）：
#   canary #1 教訓——nir-leaves −0.106 來自「真實快速運動誤觸發 → 凍結 73 幀 →
#   memory 停更 × 外觀快變」。v2 把 E15 診斷的 switch 簽名（跳後穩住）與
#   誤觸發簽名（跳後續動）編碼進狀態機：觸發後第一幀位移仍 >0.5×thr → 立即解凍。
#   事前判準（寫死）：rednir-droneshow2 Δ≥+0.03 維持｜nir-leaves |Δ|<0.02｜
#   vis-S_jump2 |Δ|<0.01｜nir-paper_crane |Δ|<0.01｜穩定 5 支 Δ=0｜vis-droneshow2 ≥0。
#   全過→排全量；任一 fail→一次修正額度用完，gate 線結案入庫。
#
# [B-v2] mosaic 直讀擴測（120 幀冒煙 IoU 0.806 vs 假色 0.705 的延伸）：
#   rednir-drone2 全 375 幀（看假色崩壞段 mosaic 是否也崩）
#   vis-droneshow2 全 450 幀（病灶！假色 AUC 0.1009；無人機 mosaic 上 4× 大）
#   nir-redbag 前 300 幀（5×5 macro 可讀性單獨驗證）
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
GOUT="$GDRIVE/5_outputs/e25_gate_20260808"
PY3=~/sam3env/bin/python
STAMP=~/e25b_timing.txt; : > "$STAMP"
die() { echo "🚨 $*"; rclone copyto "$STAMP" "$GOUT/e25b_timing_DIED.txt" 2>/dev/null; exit 1; }
mark() { echo "$1 $(date +%s)" >> "$STAMP"; echo "⏱  $1 @ $(date +%H:%M:%S)"; }
mark B2_START

SEQROOT=$(find ~ -maxdepth 2 -type d -name "vis-droneshow2" -print -quit | xargs -r dirname)
[ -n "$SEQROOT" ] || die "val 假色根目錄找不到"

# --- [1] 拉 gate v2 代碼 ------------------------------------------------------
rclone copyto "$GDRIVE/3_src/track_t1.py" ~/track_t1.py
grep -q "cont_frac" ~/track_t1.py || die "track_t1.py 不含 gate v2（cont_frac）"

# --- [2] A-v2：on 腿重跑（off 腿復用）----------------------------------------
mark ONV2_START
$PY3 ~/track_t1.py --frames-root "$SEQROOT" --seq-list ~/canary11.txt --gt-csv ~/2026training.csv \
  --out-dir ~/e25_on_v2 --backend sam3 --sam3-ckpt ~/sam3.pt --sam3-eval --memory-gate \
  > ~/e25_on_v2.log 2>&1 || die "on-v2 腿失敗"
rclone copy ~/e25_on_v2/ "$GOUT/canary_on_v2/"
$PY3 - > ~/e25b_verdict.txt 2>&1 <<'EOF'
import pandas as pd, numpy as np, json, os

def load(p):
    df = pd.read_csv(p); df.columns = ["ID","x","y","w","h"]
    s = df["ID"].str.rsplit("_", n=1)
    df["seq"], df["fid"] = s.str[0], s.str[1].astype(int)
    return df

def auc_seq(pred, gt):
    m = gt.merge(pred, on="ID", suffixes=("_g","_p"))
    x1 = np.maximum(m.x_g, m.x_p); y1 = np.maximum(m.y_g, m.y_p)
    x2 = np.minimum(m.x_g+m.w_g, m.x_p+m.w_p); y2 = np.minimum(m.y_g+m.h_g, m.y_p+m.h_p)
    inter = np.maximum(x2-x1,0)*np.maximum(y2-y1,0)
    iou = inter/(m.w_g*m.h_g + m.w_p*m.h_p - inter)
    return float(np.mean([(iou >= t).mean() for t in np.arange(0.02, 1.001, 0.02)]))

home = os.path.expanduser("~")
gt = load(f"{home}/2026training.csv")
on = load(f"{home}/e25_on_v2/submission.csv")
off = load(f"{home}/e25_off/submission.csv")
diag = json.load(open(f"{home}/e25_on_v2/diagnostics.json"))
rows = []
for seq in sorted(on.seq.unique()):
    a_on = auc_seq(on[on.seq==seq], gt[gt.seq==seq])
    a_off = auc_seq(off[off.seq==seq], gt[gt.seq==seq])
    g = diag.get(seq, {}).get("gate", {})
    rows.append({"seq": seq, "auc_on2": round(a_on,4), "auc_off": round(a_off,4),
                 "delta": round(a_on-a_off,4), "trig": g.get("n_triggers"),
                 "frozen": g.get("frozen_frames"), "ret": len(g.get("returns",[])),
                 "abort": len(g.get("aborts",[]))})
df = pd.DataFrame(rows).sort_values("delta")
print(df.to_string(index=False))
d = {r["seq"]: r["delta"] for r in rows}
checks = {
    "rednir-droneshow2 ≥+0.03": d["rednir-droneshow2"] >= 0.03,
    "nir-leaves |Δ|<0.02": abs(d["nir-leaves"]) < 0.02,
    "vis-S_jump2 |Δ|<0.01": abs(d["vis-S_jump2"]) < 0.01,
    "nir-paper_crane |Δ|<0.01": abs(d["nir-paper_crane"]) < 0.01,
    "穩定5支 Δ=0": all(d[s] == 0 for s in
        ["nir-glass_cup","nir-redbag","rednir-glass2","vis-officefan2","vis-receipts3"]),
    "vis-droneshow2 ≥0": d["vis-droneshow2"] >= 0,
}
for k, v in checks.items(): print(f"{'✅' if v else '❌'} {k}")
allpass = all(checks.values())
print(f"\n=== gate v2 判決: {'全過 → 排全量' if allpass else '未全過 → gate 線結案'} ===")
json.dump({"rows": rows, "checks": {k: bool(v) for k, v in checks.items()},
           "allpass": bool(allpass)}, open(f"{home}/e25b_verdict.json","w"), indent=1)
EOF
cat ~/e25b_verdict.txt
rclone copyto ~/e25b_verdict.txt "$GOUT/canary_v2_verdict.txt"
rclone copyto ~/e25b_verdict.json "$GOUT/canary_v2_verdict.json" 2>/dev/null || true
mark ONV2_DONE

# --- [3] B-v2：mosaic 三支 ----------------------------------------------------
mark BLINE2_START
rclone copy "$GDRIVE/1_data/raw_archive/training/HSI-VIS/droneshow2.zip" ~/mz_vis/ &
V=$!
rclone copy "$GDRIVE/1_data/raw_archive/training/HSI-NIR/redbag.zip" ~/mz_nir/ &
N=$!
wait $V $N

mos_prep() {  # $1=zip dir  $2=seq 名  $3=macro 倍率  $4=幀數上限(0=全)
  local zdir=$1 seq=$2 macro=$3 cap=$4
  local z; z=$(find "$zdir" -name "*.zip" -print -quit); [ -n "$z" ] || { echo "no zip in $zdir"; return 1; }
  rm -rf ~/mprep_raw; unzip -o -q "$z" -d ~/mprep_raw/
  MDIR=$(find ~/mprep_raw -type d -name "*" -exec sh -c 'ls "$1"/*.png >/dev/null 2>&1 && echo "$1"' _ {} \; | head -1)
  [ -n "$MDIR" ] || { echo "no png dir"; return 1; }
  $PY3 - "$MDIR" "$seq" "$macro" "$cap" <<'PYEOF'
import numpy as np, os, glob, sys
from PIL import Image
mdir, seq, macro, cap = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
files = sorted(glob.glob(os.path.join(mdir, "*.png")))
if cap: files = files[:cap]
home = os.path.expanduser("~")
out = f"{home}/bline2/frames/{seq}"
os.makedirs(out, exist_ok=True)
for i, f in enumerate(files):
    a = np.array(Image.open(f))
    assert a.dtype == np.uint16, (f, a.dtype)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    g = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    Image.merge("RGB", [Image.fromarray(g)] * 3).save(f"{out}/{i+1:04d}.jpg", quality=95)
# init = GT 首列 × macro
import pandas as pd
gt = pd.read_csv(f"{home}/2026training.csv"); gt.columns = ["ID","x","y","w","h"]
s = gt["ID"].str.rsplit("_", n=1); gt["seq"] = s.str[0]
r = gt[gt.seq == seq].iloc[0]
open(f"{out}/init_rect.txt","w").write(f"{r.x*macro} {r.y*macro} {r.w*macro} {r.h*macro}")
print(f"{seq}: {len(files)} 幀 → jpg + init(×{macro})")
PYEOF
}

mkdir -p ~/bline2/frames
mos_prep ~/mosaic_upd rednir-drone2 4 0 || mos_prep ~/mosaic rednir-drone2 4 0 || die "drone2 prep 失敗"
mos_prep ~/mz_vis vis-droneshow2 4 0 || die "vis-droneshow2 prep 失敗"
mos_prep ~/mz_nir nir-redbag 5 300 || die "nir-redbag prep 失敗"
printf "rednir-drone2\nvis-droneshow2\nnir-redbag\n" > ~/bline2_list.txt

$PY3 ~/track_t1.py --frames-root ~/bline2/frames --seq-list ~/bline2_list.txt \
  --out-dir ~/bline2/track --backend sam3 --sam3-ckpt ~/sam3.pt --sam3-eval \
  > ~/bline2/track.log 2>&1 || die "B-v2 追蹤失敗"
$PY3 - > ~/bline2/verdict.txt 2>&1 <<'EOF'
import pandas as pd, numpy as np, os, json
home = os.path.expanduser("~")
MACRO = {"rednir-drone2": 4, "vis-droneshow2": 4, "nir-redbag": 5}

def load(p):
    df = pd.read_csv(p); df.columns = ["ID","x","y","w","h"]
    s = df["ID"].str.rsplit("_", n=1)
    df["seq"], df["fid"] = s.str[0], s.str[1].astype(int)
    return df

gt = load(f"{home}/2026training.csv")
pred = load(f"{home}/bline2/track/submission.csv")
off = load(f"{home}/e25_off/submission.csv")  # 假色對照（canary off 腿）
out = {}
for seq, mac in MACRO.items():
    p = pred[pred.seq==seq].sort_values("fid").reset_index(drop=True)
    g = gt[gt.seq==seq].sort_values("fid").head(len(p)).reset_index(drop=True)
    gx, gy, gw, gh = g.x*mac, g.y*mac, g.w*mac, g.h*mac
    x1=np.maximum(gx,p.x); y1=np.maximum(gy,p.y)
    x2=np.minimum(gx+gw,p.x+p.w); y2=np.minimum(gy+gh,p.y+p.h)
    inter=np.maximum(x2-x1,0)*np.maximum(y2-y1,0)
    iou_m = inter/(gw*gh + p.w*p.h - inter)
    # 假色同窗對照
    o = off[off.seq==seq].sort_values("fid").head(len(p)).reset_index(drop=True)
    x1=np.maximum(g.x,o.x); y1=np.maximum(g.y,o.y)
    x2=np.minimum(g.x+g.w,o.x+o.w); y2=np.minimum(g.y+g.h,o.y+o.h)
    inter=np.maximum(x2-x1,0)*np.maximum(y2-y1,0)
    iou_f = inter/(g.w*g.h + o.w*o.h - inter)
    thr = np.arange(0.02, 1.001, 0.02)
    auc_m = float(np.mean([(iou_m>=t).mean() for t in thr]))
    auc_f = float(np.mean([(iou_f>=t).mean() for t in thr]))
    out[seq] = {"n": len(p), "auc_mosaic": round(auc_m,4), "auc_falsecolor": round(auc_f,4),
                "delta": round(auc_m-auc_f,4),
                "iou_med_mosaic": round(float(np.median(iou_m)),3),
                "iou_med_fc": round(float(np.median(iou_f)),3)}
    print(f"{seq}: mosaic AUC {auc_m:.4f} vs 假色 {auc_f:.4f} (Δ{auc_m-auc_f:+.4f}) "
          f"| IoU中位 {np.median(iou_m):.3f} vs {np.median(iou_f):.3f} | n={len(p)}")
json.dump(out, open(f"{home}/bline2/verdict.json","w"), indent=1)
EOF
cat ~/bline2/verdict.txt
rclone copyto ~/bline2/verdict.txt "$GOUT/bline2_verdict.txt"
rclone copyto ~/bline2/verdict.json "$GOUT/bline2_verdict.json" 2>/dev/null || true
rclone copy ~/bline2/track/ "$GOUT/bline2_track/" 2>/dev/null || true
mark BLINE2_DONE

rclone copyto "$STAMP" "$GOUT/e25b_timing.txt"
mark B2_ALL_DONE
echo "✅ e25b 完成。機器保留（保險自毀仍在）。"
