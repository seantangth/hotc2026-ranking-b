#!/usr/bin/env bash
# ============================================================================
# setup_e25_gate_canary_v1.sh — 08-08 三線驗證（一台 A100，全自動非互動）
#
#   [D] S-04 eval-mode 探針：tracker.training 實測 + train/eval 單序列 A/B diff
#       （上游 build_sam3_video_model 漏 .eval()，memory attention dropout=0.1 推定生效）
#   [A] E25 anti-switch memory gate canary：11 支 × 兩腿（gate off/on，皆 --sam3-eval）
#       病灶跳變型 ×2（vis-droneshow2/rednir-droneshow2，離線前測 switch 位移 80-98px
#       vs GT 真實運動 ≤2.7px/幀）＋漂移型 ×1（rednir-drone2，無預期、觀察）
#       ＋E24 穩定組 ×5 ＋離線誤觸發前 3（nir-leaves/vis-S_jump2/nir-paper_crane）
#       ⚠️ 主判定＝on−off（同 mode 同 RNG 域）。E15 基準只當脈絡——若 S-04 屬實，
#       E15 是 train-mode+65 支 RNG 流的產物，與本次 11 支不可比（S-08 機制）。
#       兩腿皆 --sam3-eval：決定性（僅剩 CUDA 噪聲 ~0.0002 級），且 eval 正是
#       S-04 屬實時的未來部署 mode——在正確的 regime 裡評 gate。
#   [B] raw mosaic 直讀前測：rednir-drone2 uint16 mosaic → 灰階 jpg → SAM3 冒煙
#       （真實解析度 ×4 零插值；與 E20 插值偽影失敗機制正交）
#
# 【事前判準（寫死，跑完照著判）】
#   A-機制成功：droneshow2 兩支至少一支 AUC +0.03↑；A-安全：穩定 5 支 |Δ|<0.005、
#   誤觸發 3 支 |Δ|<0.01。D-有效：train/eval diff 幀 >1% 即證 dropout 實際影響輸出。
#   B-可讀：mosaic 冒煙 50 幀 mask 非空率 >80% 且中位 IoU(vs GT) >0.3。
#
# 【紀律】D036 單 tar｜D018 保險自毀+用完 terminate｜D016 每階段完成立即 rclone｜
#        兩套計時器盤點（pgrep -af "sleep [0-9]+"）｜venv 完全隔離｜檔名帶版本號
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
GOUT="$GDRIVE/5_outputs/e25_gate_20260808"
PY3=~/sam3env/bin/python
INSTANCE_NAME="${INSTANCE_NAME:-hsot-e25}"
STAMP=~/e25_timing.txt; : > "$STAMP"
die() { echo "🚨 $*"; rclone copyto "$STAMP" "$GOUT/timing_DIED.txt" 2>/dev/null; exit 1; }
mark() { echo "$1 $(date +%s)" >> "$STAMP"; echo "⏱  $1 @ $(date +%H:%M:%S)"; }
mark START

# --- 保險自毀 4h（D018；延長死線須先 pgrep -af "sleep [0-9]+" 盤點兩套計時器）----
nohup bash -c "
  sleep 14400
  key=\$(cat ~/.lambda_key | tr -d '\n'); [ -z \"\$key\" ] && exit 0
  id=\$(curl -s -u \"\$key:\" https://cloud.lambda.ai/api/v1/instances | python3 -c \"
import json,sys
d=json.load(sys.stdin).get('data',[])
m=[i['id'] for i in d if i.get('name')=='$INSTANCE_NAME']
print(m[0] if m else '')\")
  [ -n \"\$id\" ] && curl -s -u \"\$key:\" -X POST \
    https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H 'Content-Type: application/json' -d \"{\\\"instance_ids\\\":[\\\"\$id\\\"]}\"
" > ~/insurance_destruct.log 2>&1 &
echo "⏰ 4h 保險自毀已掛（$INSTANCE_NAME）"

# --- [1] 資料 ‖ 環境 並行（D036）--------------------------------------------
mark ENV_DATA_START
(
  rclone copy "$GDRIVE/1_data/packed/t1val_fc_65.tar" ~/ && tar -xf ~/t1val_fc_65.tar -C ~/
  rclone copyto "$GDRIVE/1_data/raw/2026training.csv" ~/2026training.csv
  # B 前測資料（1GB mosaic + 官方 update 補包若存在）
  rclone copy "$GDRIVE/1_data/raw_archive/training/HSI-RedNIR/drone2.zip" ~/mosaic/ || true
  rclone copy "$GDRIVE/1_data/raw_archive/training/update/HSI-RedNIR/drone2.zip" ~/mosaic_upd/ || true
  echo DATA_READY
) > ~/data.log 2>&1 &
DATA_PID=$!

# sam3env（PROVENANCE 四坑配方；setuptools 最後降級）
[ -d ~/sam3env ] || uv venv --python 3.12 ~/sam3env
if [ ! -f ~/sam3env/.deps_done ]; then
  VIRTUAL_ENV=~/sam3env uv pip install -q torch torchvision --torch-backend=auto || die "torch 安裝失敗"
  VIRTUAL_ENV=~/sam3env uv pip install -q "git+https://github.com/facebookresearch/sam3.git" \
    "numpy<2" ftfy==6.1.1 regex einops psutil scipy av \
    pycocotools numba python-rapidjson pandas pillow tqdm opencv-python-headless || die "sam3 安裝失敗"
  VIRTUAL_ENV=~/sam3env uv pip install -q "setuptools<81" || die "setuptools 降級失敗"
  touch ~/sam3env/.deps_done || die "deps_done 標記失敗（S-05：不可 fail-open，否則同機 resume 永不自我修復）"
fi
$PY3 -c "
import sam3, numpy, pkg_resources, torch
assert torch.cuda.is_available(), 'CUDA 不可用'
assert numpy.__version__.startswith('1.'), f'numpy {numpy.__version__} 應 <2'
print(f'sam3env OK: torch {torch.__version__} numpy {numpy.__version__}')
" || die "sam3env 驗證失敗"

# 權重（現行鏡像配方；官方 gated 換裝為封板前任務 D041）
[ -f ~/sam3.pt ] || $PY3 - <<'EOF' || die "權重下載失敗"
from huggingface_hub import hf_hub_download
import os, shutil
p = hf_hub_download("1038lab/sam3", "sam3.pt")
shutil.copy(p, os.path.expanduser("~/sam3.pt"))
print("ckpt OK")
EOF
[ -f ~/sam3.pt ] || die "sam3.pt 不存在"

# 程式碼（本次 push 的新版：E25 gate + --sam3-eval）
rclone copyto "$GDRIVE/3_src/track_t1.py" ~/track_t1.py
rclone copy "$GDRIVE/3_src/hsot/" ~/hsot/ --include "*.py"
grep -q "memory_gate" ~/track_t1.py || die "track_t1.py 不含 E25 gate——gDrive 版本未更新"
wait $DATA_PID; grep -q DATA_READY ~/data.log || die "資料就緒失敗（見 ~/data.log）"
SEQROOT=$(find ~ -maxdepth 2 -type d -name "t1val_fc_65" -print -quit)
[ -n "$SEQROOT" ] || SEQROOT=~/t1val_fc_65
[ -d "$SEQROOT/vis-droneshow2" ] || SEQROOT=$(find ~ -maxdepth 3 -type d -name "vis-droneshow2" -print -quit | xargs -r dirname)
[ -d "$SEQROOT/vis-droneshow2" ] || die "val 假色目錄結構異常"
mark ENV_DATA_DONE

# --- [2] D 線：S-04 探針（flag 實測 + train/eval 單序列 A/B）-------------------
mark PROBE_START
mkdir -p ~/probe
$PY3 - > ~/probe/s04_probe.txt 2>&1 <<EOF
import torch, json
from sam3.model_builder import build_sam3_video_model
m = build_sam3_video_model(device="cuda:0", checkpoint_path="$HOME/sam3.pt")
tr = m.tracker
n_train = sum(1 for mod in m.modules() if mod.training)
n_total = sum(1 for _ in m.modules())
drops = {name: mod.p for name, mod in m.named_modules() if isinstance(mod, torch.nn.Dropout) and mod.p > 0}
# functional dropout：RoPEAttention 把 dropout_p 存成 float 屬性、直接餵
# F.scaled_dot_product_attention——不是 nn.Dropout 子模組，isinstance 掃不到
fdrops = {name: getattr(mod, "dropout_p", 0) for name, mod in m.named_modules()
          if getattr(mod, "dropout_p", 0) and not isinstance(mod, torch.nn.Dropout)}
report = {"model_training": m.training, "tracker_training": tr.training,
          "modules_in_train_mode": f"{n_train}/{n_total}",
          "nn_dropout_modules": len(drops),
          "functional_dropout_attrs": len(fdrops),
          "sample_functional": dict(list(fdrops.items())[:10]),
          "sample_dropouts": dict(list(drops.items())[:8])}
print(json.dumps(report, indent=1))
open("$HOME/probe/s04_flags.json", "w").write(json.dumps(report))
EOF
cat ~/probe/s04_probe.txt
rclone copy ~/probe/ "$GOUT/probe/"

# train vs eval 的實際輸出 diff（同序列前 80 幀跑兩遍；diff>1% 即 dropout 實際影響輸出）
SHORTSEQ=rednir-glass2
mkdir -p ~/probe_ab/train ~/probe_ab/eval ~/ab_frames/$SHORTSEQ
for f in $(find "$SEQROOT/$SHORTSEQ" -name "*.jpg" | sort | sed -n '1,80p'); do cp "$f" ~/ab_frames/$SHORTSEQ/; done
echo "$SHORTSEQ" > ~/ab_list.txt
$PY3 ~/track_t1.py --frames-root ~/ab_frames --seq-list ~/ab_list.txt --gt-csv ~/2026training.csv \
  --out-dir ~/probe_ab/train --backend sam3 --sam3-ckpt ~/sam3.pt > ~/probe_ab/train.log 2>&1 || die "A/B train 腿失敗"
$PY3 ~/track_t1.py --frames-root ~/ab_frames --seq-list ~/ab_list.txt --gt-csv ~/2026training.csv \
  --out-dir ~/probe_ab/eval --backend sam3 --sam3-ckpt ~/sam3.pt --sam3-eval > ~/probe_ab/eval.log 2>&1 || die "A/B eval 腿失敗"
$PY3 - > ~/probe/s04_ab_diff.txt 2>&1 <<EOF
import pandas as pd, numpy as np, json
a = pd.read_csv("$HOME/probe_ab/train/submission.csv"); b = pd.read_csv("$HOME/probe_ab/eval/submission.csv")
m = a.merge(b, on="ID", suffixes=("_t", "_e"))
diff = ((m.x_t != m.x_e) | (m.y_t != m.y_e) | (m.width_t != m.width_e) | (m.height_t != m.height_e))
cd = np.hypot(m.x_t + m.width_t/2 - m.x_e - m.width_e/2, m.y_t + m.height_t/2 - m.y_e - m.height_e/2)
r = {"n": len(m), "diff_frames": int(diff.sum()), "diff_pct": round(100*diff.mean(), 1),
     "center_shift_median": float(np.median(cd)), "center_shift_max": float(cd.max())}
print(json.dumps(r)); open("$HOME/probe/s04_ab.json", "w").write(json.dumps(r))
EOF
cat ~/probe/s04_ab_diff.txt
rclone copy ~/probe/ "$GOUT/probe/"
mark PROBE_DONE

# --- [3] A 線：E25 gate canary 11 支 ----------------------------------------
mark CANARY_START
cat > ~/canary11.txt <<'LIST'
vis-droneshow2
rednir-droneshow2
rednir-drone2
nir-glass_cup
nir-redbag
rednir-glass2
vis-officefan2
vis-receipts3
nir-leaves
vis-S_jump2
nir-paper_crane
LIST
# 兩腿同 mode 同 RNG 域：off 腿=乾淨基準；on 腿=唯一變因 gate（皆 --sam3-eval）
$PY3 ~/track_t1.py --frames-root "$SEQROOT" --seq-list ~/canary11.txt --gt-csv ~/2026training.csv \
  --out-dir ~/e25_off --backend sam3 --sam3-ckpt ~/sam3.pt --sam3-eval \
  > ~/e25_off.log 2>&1 || die "off 腿失敗（見 ~/e25_off.log）"
rclone copy ~/e25_off/ "$GOUT/canary_off/"
$PY3 ~/track_t1.py --frames-root "$SEQROOT" --seq-list ~/canary11.txt --gt-csv ~/2026training.csv \
  --out-dir ~/e25_on --backend sam3 --sam3-ckpt ~/sam3.pt --sam3-eval --memory-gate \
  > ~/e25_on.log 2>&1 || die "on 腿失敗（見 ~/e25_on.log）"
tail -15 ~/e25_on.log
rclone copy ~/e25_on/ "$GOUT/canary_on/"

# canary 評分：主判定 = on−off；E15 欄僅脈絡（含 mode/RNG 差異，不可當判準）
rclone copyto "$GDRIVE/5_outputs/e15_sam3_20260806/submission_val65.csv" ~/e15_val65.csv || true
$PY3 - > ~/e25_verdict.txt 2>&1 <<'EOF'
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
    thr = np.arange(0.02, 1.001, 0.02)
    return float(np.mean([(iou >= t).mean() for t in thr]))

home = os.path.expanduser("~")
gt = load(f"{home}/2026training.csv")
on = load(f"{home}/e25_on/submission.csv")
off = load(f"{home}/e25_off/submission.csv")
e15 = load(f"{home}/e15_val65.csv") if os.path.exists(f"{home}/e15_val65.csv") else None
diag = json.load(open(f"{home}/e25_on/diagnostics.json"))
rows = []
for seq in sorted(on.seq.unique()):
    a_on = auc_seq(on[on.seq==seq], gt[gt.seq==seq])
    a_off = auc_seq(off[off.seq==seq], gt[gt.seq==seq])
    a_e15 = auc_seq(e15[e15.seq==seq], gt[gt.seq==seq]) if e15 is not None else float("nan")
    g = diag.get(seq, {}).get("gate", {})
    rows.append({"seq": seq, "auc_on": round(a_on,4), "auc_off": round(a_off,4),
                 "delta_on_off": round(a_on-a_off,4),
                 "e15_ctx": round(a_e15,4),
                 "off_vs_e15": round(a_off-a_e15,4),  # S-04/S-08 的旁證：≠0 即 mode/RNG 差異實錘
                 "n_trig": g.get("n_triggers"), "frozen": g.get("frozen_frames"),
                 "returns": len(g.get("returns",[]))})
df = pd.DataFrame(rows).sort_values("delta_on_off")
print(df.to_string(index=False))
PATHO_JUMP = {"vis-droneshow2","rednir-droneshow2"}
STABLE = {"nir-glass_cup","nir-redbag","rednir-glass2","vis-officefan2","vis-receipts3"}
FT = {"nir-leaves","vis-S_jump2","nir-paper_crane"}
d = {r["seq"]: r["delta_on_off"] for r in rows}
mech = any(d[s] > 0.03 for s in PATHO_JUMP)
safe = all(abs(d[s]) < 0.005 for s in STABLE) and all(abs(d[s]) < 0.01 for s in FT)
print(f"\n機制驗證(droneshow2 任一 on−off +0.03↑): {'✅' if mech else '❌'}")
print(f"安全驗證(穩定|Δ|<0.005, 誤觸發|Δ|<0.01): {'✅' if safe else '❌'}")
off_e15 = [r["off_vs_e15"] for r in rows if not np.isnan(r["off_vs_e15"])]
if off_e15:
    print(f"off vs E15 差異（S-04 旁證）：中位 {np.median(np.abs(off_e15)):.4f} 最大 {np.max(np.abs(off_e15)):.4f}")
json.dump({"rows": rows, "mech": bool(mech), "safe": bool(safe)},
          open(f"{home}/e25_verdict.json","w"), indent=1)
EOF
cat ~/e25_verdict.txt
rclone copyto ~/e25_verdict.txt "$GOUT/canary_verdict.txt"
rclone copyto ~/e25_verdict.json "$GOUT/canary_verdict.json" 2>/dev/null || true
mark CANARY_DONE

# --- [4] B 線：mosaic 直讀前測 ------------------------------------------------
mark MOSAIC_START
mkdir -p ~/bline
MZIP=$(find ~/mosaic_upd ~/mosaic -name "drone2.zip" -print -quit 2>/dev/null)
if [ -n "$MZIP" ]; then
  unzip -o -q "$MZIP" -d ~/bline/raw/
  MDIR=$(find ~/bline/raw -type d -name "*drone2*" -print -quit); [ -n "$MDIR" ] || MDIR=~/bline/raw
  $PY3 - > ~/bline/mosaic_probe.txt 2>&1 <<EOF
import numpy as np, os, glob, json
from PIL import Image
mdir = "$MDIR"
files = sorted(glob.glob(os.path.join(mdir, "**", "*.png"), recursive=True))
print(f"mosaic png: {len(files)}")
os.makedirs("$HOME/bline/gray", exist_ok=True)
os.makedirs("$HOME/bline/frames/rednir-drone2", exist_ok=True)
stats = []
for i, f in enumerate(files):
    a = np.array(Image.open(f))
    assert a.dtype == np.uint16, a.dtype
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    g = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    im = Image.merge("RGB", [Image.fromarray(g)] * 3)
    im.save(f"$HOME/bline/frames/rednir-drone2/{i+1:04d}.jpg", quality=95)
    if i < 3:
        im.save(f"$HOME/bline/gray/probe_{i+1:04d}.jpg", quality=95)
        stats.append({"f": os.path.basename(f), "shape": list(a.shape), "p1": float(lo), "p99": float(hi)})
print(json.dumps(stats, indent=1))
EOF
  cat ~/bline/mosaic_probe.txt
  rclone copy ~/bline/gray/ "$GOUT/bline_vis/"
  # SAM3 冒煙：mosaic 域 GT init（cube 座標 ×4），前 120 幀
  $PY3 - <<'EOF'
import pandas as pd, os
home = os.path.expanduser("~")
gt = pd.read_csv(f"{home}/2026training.csv"); gt.columns = ["ID","x","y","w","h"]
s = gt["ID"].str.rsplit("_", n=1); gt["seq"] = s.str[0]
r = gt[gt.seq == "rednir-drone2"].iloc[0]
os.makedirs(f"{home}/bline/frames/rednir-drone2", exist_ok=True)
open(f"{home}/bline/frames/rednir-drone2/init_rect.txt","w").write(
    f"{r.x*4} {r.y*4} {r.w*4} {r.h*4}")
open(f"{home}/bline_list.txt","w").write("rednir-drone2")
print("init(mosaic 座標):", r.x*4, r.y*4, r.w*4, r.h*4)
EOF
  # 只留前 120 幀省時（tail/xargs 皆讀完輸入，無 SIGPIPE 風險）
  find ~/bline/frames/rednir-drone2 -name "*.jpg" | sort | tail -n +121 | xargs -r rm
  $PY3 ~/track_t1.py --frames-root ~/bline/frames --seq-list ~/bline_list.txt \
    --out-dir ~/bline/track --backend sam3 --sam3-ckpt ~/sam3.pt \
    > ~/bline/track.log 2>&1 || echo "B 冒煙失敗（非致命）"
  $PY3 - > ~/bline/verdict.txt 2>&1 <<'EOF'
import pandas as pd, numpy as np, os
home = os.path.expanduser("~")
p = f"{home}/bline/track/submission.csv"
if os.path.exists(p):
    pred = pd.read_csv(p); pred.columns = ["ID","x","y","w","h"]
    s = pred["ID"].str.rsplit("_", n=1); pred["fid"] = s.str[1].astype(int)
    gt = pd.read_csv(f"{home}/2026training.csv"); gt.columns = ["ID","x","y","w","h"]
    s = gt["ID"].str.rsplit("_", n=1); gt["seq"], gt["fid"] = s.str[0], s.str[1].astype(int)
    g = gt[gt.seq == "rednir-drone2"].sort_values("fid").head(len(pred)).reset_index(drop=True)
    pred = pred.sort_values("fid").reset_index(drop=True)
    gx, gy, gw, gh = g.x*4, g.y*4, g.w*4, g.h*4  # GT → mosaic 座標
    x1 = np.maximum(gx, pred.x); y1 = np.maximum(gy, pred.y)
    x2 = np.minimum(gx+gw, pred.x+pred.w); y2 = np.minimum(gy+gh, pred.y+pred.h)
    inter = np.maximum(x2-x1,0)*np.maximum(y2-y1,0)
    iou = inter/(gw*gh + pred.w*pred.h - inter)
    print(f"B 冒煙 {len(pred)} 幀: IoU 中位 {np.median(iou):.3f} 平均 {iou.mean():.3f} "
          f">0.3 比例 {(iou>0.3).mean():.1%}")
    print("判準(可讀性): 中位 IoU>0.3 →", "✅" if np.median(iou) > 0.3 else "❌")
    print("⚠️ 期望值校準：mosaic 2048×1088 → SAM3 內部 1008² 是 2× downscale，")
    print("   模型輸入的像素預算與假色路徑幾乎相同——×16 真實像素只有配 crop-zoom")
    print("   （窗取在 mosaic 座標）才真正到達模型。本冒煙 pass 只證明『SAM3 讀得懂")
    print("   mosaic 紋理』→ 下一步是 mosaic×crop 組合，不是 mosaic 單獨部署。")
else:
    print("B 冒煙無輸出")
EOF
  cat ~/bline/verdict.txt
  rclone copy ~/bline/verdict.txt "$GOUT/bline_vis/" 2>/dev/null || true
  rclone copy ~/bline/track/ "$GOUT/bline_track/" 2>/dev/null || true
  rclone copy ~/bline/mosaic_probe.txt "$GOUT/bline_vis/" 2>/dev/null || true
else
  echo "⚠️ mosaic zip 未就緒，B 線跳過"
fi
mark MOSAIC_DONE

rclone copyto "$STAMP" "$GOUT/timing.txt"
mark ALL_DONE
echo "=========================================="
echo "✅ 全部完成。產出已在 $GOUT"
echo "機器保留中（4h 保險自毀仍在）——依 verdict 決定是否接跑 E26 全量 eval 探針"
echo "=========================================="
