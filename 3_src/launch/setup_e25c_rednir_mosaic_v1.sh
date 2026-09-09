#!/usr/bin/env bash
# ============================================================================
# setup_e25c_rednir_mosaic_v1.sh — 接力：val rednir 11 支 mosaic 直讀全量
#
# 依據（e25b B 線）：rednir-drone2 mosaic AUC 0.7431 vs 假色 0.3414（+0.40）——
#   RedNIR 低照度（p99≈708/65535）下官方假色 tone mapping 壓掉對比；
#   mosaic 直讀（p1-p99 正規化 ×4 真實解析度）救回。nir-redbag −0.008（5×5 可讀、
#   高分序列打平）；vis-droneshow2 +0.004（switch 病灶對解析度不敏感）。
#
# 部署形態（Ranking B 合法）：模態=rednir → mosaic 直讀；vis/nir → 假色不動。
#   客觀屬性、不認序列名、逐幀因果轉換。
#
# 事前判準（D038/D040）：
#   假色對照 = E15 submission_val65.csv（off_vs_e15=0.0000 已證同 mode 同輸出）。
#   (a) 11 支 pooled Δ>+0.01 且 (b) 最壞單支崩幅 >−0.05 者 0 支 → 排 test 提交候選；
#   混合訊號 → 記錄逐序列 delta 供對比閾值篩選規則設計；淨負 → 結案。
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
GOUT="$GDRIVE/5_outputs/e25_gate_20260808"
RAWT="$GDRIVE/1_data/raw_archive/training"
PY3=~/sam3env/bin/python
STAMP=~/e25c_timing.txt; : > "$STAMP"
die() { echo "🚨 $*"; rclone copyto "$STAMP" "$GOUT/e25c_timing_DIED.txt" 2>/dev/null; exit 1; }
mark() { echo "$1 $(date +%s)" >> "$STAMP"; echo "⏱  $1 @ $(date +%H:%M:%S)"; }
mark C_START

SEQS="rednir-backpack3 rednir-backpack4 rednir-bytheriver1 rednir-drone2 rednir-drone4 rednir-droneshow2 rednir-droneshow4 rednir-foot1 rednir-glass2 rednir-officefan2 rednir-receipts3"

# --- [1] 下載 11 支 mosaic（update 補包優先）並行 4 路 -------------------------
mark DL_START
mkdir -p ~/mz_c
dl_one() {
  local seq=$1 base=${1#rednir-}
  [ -d ~/mz_c/"$seq" ] && return 0
  mkdir -p ~/mz_c/"$seq"
  rclone copy "$RAWT/update/HSI-RedNIR/${base}.zip" ~/mz_c/"$seq"/ 2>/dev/null \
    || rclone copy "$RAWT/HSI-RedNIR/${base}.zip" ~/mz_c/"$seq"/ \
    || { echo "MISS $seq"; return 1; }
}
export -f dl_one; export RAWT
echo "$SEQS" | tr ' ' '\n' | xargs -P 4 -I{} bash -c 'dl_one {}' || die "下載失敗"
mark DL_DONE

# --- [2] 轉檔（uint16 mosaic → p1-p99 灰階 jpg ×3ch）＋ init ×4 ---------------
mark PREP_START
mkdir -p ~/blinec/frames
for seq in $SEQS; do
  [ -d ~/blinec/frames/"$seq" ] && continue
  Z=$(find ~/mz_c/"$seq" -name "*.zip" -print -quit); [ -n "$Z" ] || die "$seq zip 缺"
  rm -rf ~/mprep_c; unzip -o -q "$Z" -d ~/mprep_c/
  MDIR=$(find ~/mprep_c -type d -exec sh -c 'ls "$1"/*.png >/dev/null 2>&1 && echo "$1"' _ {} \; | head -1)
  [ -n "$MDIR" ] || die "$seq png 目錄缺"
  $PY3 - "$MDIR" "$seq" <<'PYEOF' || exit 1
import numpy as np, os, glob, sys
from PIL import Image
mdir, seq = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(mdir, "*.png")))
assert files, mdir
home = os.path.expanduser("~")
out = f"{home}/blinec/frames/{seq}"
os.makedirs(out, exist_ok=True)
for i, f in enumerate(files):
    a = np.array(Image.open(f))
    assert a.dtype == np.uint16, (f, a.dtype)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    g = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    Image.merge("RGB", [Image.fromarray(g)] * 3).save(f"{out}/{i+1:04d}.jpg", quality=95)
import pandas as pd
gt = pd.read_csv(f"{home}/2026training.csv"); gt.columns = ["ID","x","y","w","h"]
s = gt["ID"].str.rsplit("_", n=1); gt["seq"] = s.str[0]
r = gt[gt.seq == seq].iloc[0]
open(f"{out}/init_rect.txt","w").write(f"{r.x*4} {r.y*4} {r.w*4} {r.h*4}")
print(f"{seq}: {len(files)} 幀 OK")
PYEOF
  echo "$seq prep done"
done
mark PREP_DONE

# --- [3] track 11 支（eval mode）---------------------------------------------
mark TRACK_START
echo "$SEQS" | tr ' ' '\n' > ~/blinec_list.txt
$PY3 ~/track_t1.py --frames-root ~/blinec/frames --seq-list ~/blinec_list.txt \
  --out-dir ~/blinec/track --backend sam3 --sam3-ckpt ~/sam3.pt --sam3-eval \
  > ~/blinec/track.log 2>&1 || die "track 失敗"
tail -13 ~/blinec/track.log
rclone copy ~/blinec/track/ "$GOUT/blinec_track/"
mark TRACK_DONE

# --- [4] verdict：mosaic vs E15 假色（AUC 座標各自原生，GT ×4 對 mosaic）------
$PY3 - > ~/blinec/verdict.txt 2>&1 <<'EOF'
import pandas as pd, numpy as np, os, json
home = os.path.expanduser("~")

def load(p):
    df = pd.read_csv(p); df.columns = ["ID","x","y","w","h"]
    s = df["ID"].str.rsplit("_", n=1)
    df["seq"], df["fid"] = s.str[0], s.str[1].astype(int)
    return df

def auc(iou):
    return float(np.mean([(iou >= t).mean() for t in np.arange(0.02, 1.001, 0.02)]))

def iou_of(g, p, scale=1):
    gx, gy, gw, gh = g.x*scale, g.y*scale, g.w*scale, g.h*scale
    x1 = np.maximum(gx, p.x); y1 = np.maximum(gy, p.y)
    x2 = np.minimum(gx+gw, p.x+p.w); y2 = np.minimum(gy+gh, p.y+p.h)
    inter = np.maximum(x2-x1, 0)*np.maximum(y2-y1, 0)
    return inter/(gw*gh + p.w*p.h - inter)

gt = load(f"{home}/2026training.csv")
mos = load(f"{home}/blinec/track/submission.csv")
e15 = load(f"{home}/e15_val65.csv")
rows, ious_m, ious_f = [], [], []
for seq in sorted(mos.seq.unique()):
    g = gt[gt.seq==seq].sort_values("fid").reset_index(drop=True)
    m = mos[mos.seq==seq].sort_values("fid").reset_index(drop=True)
    f = e15[e15.seq==seq].sort_values("fid").reset_index(drop=True)
    n = min(len(g), len(m), len(f))
    im = iou_of(g.head(n), m.head(n), 4); iff = iou_of(g.head(n), f.head(n), 1)
    ious_m.append(im); ious_f.append(iff)
    rows.append({"seq": seq, "n": n, "auc_mosaic": round(auc(im),4),
                 "auc_fc": round(auc(iff),4), "delta": round(auc(im)-auc(iff),4)})
df = pd.DataFrame(rows).sort_values("delta")
print(df.to_string(index=False))
pm = auc(np.concatenate(ious_m)); pf = auc(np.concatenate(ious_f))
deltas = df.delta.to_numpy()
print(f"\npooled(11 支): mosaic {pm:.4f} vs 假色 {pf:.4f} → Δ {pm-pf:+.4f}")
print(f"改善 {(deltas>0.005).sum()} / 退步 {(deltas<-0.005).sum()} / 持平 {((deltas>=-0.005)&(deltas<=0.005)).sum()}")
print(f"最壞單支 {deltas.min():+.4f}（{df.iloc[0].seq}）｜最佳 {deltas.max():+.4f}（{df.iloc[-1].seq}）")
ok_pool = (pm - pf) > 0.01
ok_worst = deltas.min() > -0.05
print(f"\n判準A pooled>+0.01: {'✅' if ok_pool else '❌'}   判準B 最壞>−0.05: {'✅' if ok_worst else '❌'}")
print("→", "排 test 提交候選（rednir 換 mosaic）" if ok_pool and ok_worst else "看逐序列訊號設計篩選規則或結案")
json.dump({"rows": rows, "pooled_mosaic": pm, "pooled_fc": pf,
           "pass": bool(ok_pool and ok_worst)},
          open(f"{home}/blinec/verdict.json","w"), indent=1)
EOF
cat ~/blinec/verdict.txt
rclone copyto ~/blinec/verdict.txt "$GOUT/blinec_verdict.txt"
rclone copyto ~/blinec/verdict.json "$GOUT/blinec_verdict.json" 2>/dev/null || true
rclone copyto "$STAMP" "$GOUT/e25c_timing.txt"
mark C_ALL_DONE
echo "✅ e25c 完成"
