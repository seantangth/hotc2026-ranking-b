#!/usr/bin/env bash
# ============================================================================
# setup_e26_mosaic_crop_v1.sh — crop 窗內「感測器真實像素 vs 插值上採樣」單變因對照
#
# 假說（D046 的正交延伸，機制從未被測過）：
#   v008 的 crop-zoom 把約 100×100 的窗上採樣到 SAM3 的 1008 ＝ 插值放大 4–10 倍。
#   而官方假色的每個像素本來就是 macro×macro（實測本批全為 4×4）個**真實感測器像素**
#   壓縮而成 ⇒ 同一個窗改用 mosaic 原始像素 ＝ 把「插值猜的」換成「真的拍到的」。
#   本機 prep 實測：假色窗 228×118 → mosaic 窗 912×472，送進 1008 的插值倍率
#   由 4.4× 降到 1.1× ⇒ 插值幾乎完全消失。
#   ⚠️ 與 D046 不衝突：v014（image_size 1344 疊 crop）的 −0.0208 死因是**插值過量**，
#   真實像素與該機制正交，不受「上採樣甜蜜點」約束。
#
# 設計：兩腿共用**同一組 crop 窗**（crop_windows_val.json），同一台機器、同一次執行
#   ⇒ 唯一變因＝窗內像素來源。腿 A(fc)＝現行 v008 行為；腿 B(mosaic)＝真實像素。
#   3 支皆為跟丟重災區（E15 無 crop pooled 僅 0.1510、跟丟率 70–84%）。
#
# 事前判準（D038/D040，**寫死於執行前**）：
#   (a) 3 支中 **≥2 支** mosaic > fc，且 (b) 最壞單支 Δ > **−0.05** → 擴到全量 test
#   任一支 Δ < −0.10，或改善 <2 支 → 結案（記錄逐序列 delta 與型態）
#   健全性錨點：fc 腿 pooled 應 **> E15 無 crop 的 0.1510**（crop 本身有效才算實驗成立）
#
# 【紀律】D036 單 tar｜D018 保險自毀＋用完 terminate｜D016 每階段完成立即 rclone
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
GOUT="$GDRIVE/5_outputs/e26_mosaic_crop_20260809"
PY3=~/sam3env/bin/python
INSTANCE_NAME="${INSTANCE_NAME:-hsot-e26}"
STAMP=~/e26_timing.txt; : > "$STAMP"
die() { echo "🚨 $*"; rclone copyto "$STAMP" "$GOUT/timing_DIED.txt" 2>/dev/null; exit 1; }
mark() { echo "$1 $(date +%s)" >> "$STAMP"; echo "⏱  $1 @ $(date +%H:%M:%S)"; }
mark START

# --- 保險自毀 2h（D018；延長死線須先 pgrep -af "sleep [0-9]+" 盤點兩套計時器）----
nohup bash -c "
  sleep 7200
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
echo "⏰ 2h 保險自毀已掛（$INSTANCE_NAME）"

# --- [1] 資料 ‖ 環境 並行（D036：單一 tar，禁執行期逐檔 rclone）--------------
mark ENV_DATA_START
(
  rclone copy "$GDRIVE/1_data/packed/e26_frames.tar" ~/ && tar -xf ~/e26_frames.tar -C ~/
  rclone copyto "$GDRIVE/1_data/raw/2026training.csv" ~/2026training.csv
  echo DATA_READY
) > ~/data.log 2>&1 &
DATA_PID=$!

[ -d ~/sam3env ] || uv venv --python 3.12 ~/sam3env
if [ ! -f ~/sam3env/.deps_done ]; then
  VIRTUAL_ENV=~/sam3env uv pip install -q torch torchvision --torch-backend=auto || die "torch 安裝失敗"
  VIRTUAL_ENV=~/sam3env uv pip install -q "git+https://github.com/facebookresearch/sam3.git" \
    "numpy<2" ftfy==6.1.1 regex einops psutil scipy av \
    pycocotools numba python-rapidjson pandas pillow tqdm opencv-python-headless || die "sam3 安裝失敗"
  VIRTUAL_ENV=~/sam3env uv pip install -q "setuptools<81" || die "setuptools 降級失敗"
  touch ~/sam3env/.deps_done || die "deps_done 標記失敗"
fi
$PY3 -c "
import sam3, numpy, pkg_resources, torch
assert torch.cuda.is_available(), 'CUDA 不可用'
assert numpy.__version__.startswith('1.'), f'numpy {numpy.__version__} 應 <2'
print(f'sam3env OK: torch {torch.__version__} numpy {numpy.__version__}')
" || die "sam3env 驗證失敗"

[ -f ~/sam3.pt ] || $PY3 - <<'EOF' || die "權重下載失敗"
from huggingface_hub import hf_hub_download
import os, shutil
p = hf_hub_download("1038lab/sam3", "sam3.pt")
shutil.copy(p, os.path.expanduser("~/sam3.pt"))
print("ckpt OK")
EOF
[ -f ~/sam3.pt ] || die "sam3.pt 不存在"

rclone copyto "$GDRIVE/3_src/track_t1.py" ~/track_t1.py
grep -q "write_csv_atomic" ~/track_t1.py || die "track_t1.py 非 08-09 fail-closed 版——gDrive 未更新"
wait $DATA_PID; grep -q DATA_READY ~/data.log || die "資料就緒失敗（見 ~/data.log）"
[ -d ~/fc/vis-droneshow2 ] && [ -d ~/mosaic/vis-droneshow2 ] || die "frames 目錄結構異常"
mark ENV_DATA_DONE

# --- [2] 兩腿推論（同機同次，唯一變因＝窗內像素來源）-------------------------
printf '%s\n' vis-droneshow2 rednir-droneshow2 vis-S_jump2 > ~/e26_list.txt
for LEG in fc mosaic; do
  mark "TRACK_${LEG}_START"
  $PY3 ~/track_t1.py --frames-root ~/$LEG --seq-list ~/e26_list.txt \
    --out-dir ~/out_$LEG --backend sam3 --sam3-ckpt ~/sam3.pt --sam3-eval \
    > ~/track_$LEG.log 2>&1 || die "$LEG track 失敗（見 ~/track_$LEG.log）"
  tail -6 ~/track_$LEG.log
  rclone copy ~/out_$LEG/ "$GOUT/out_$LEG/"     # D016：每階段完成立即回傳
  rclone copyto ~/track_$LEG.log "$GOUT/track_$LEG.log"
  mark "TRACK_${LEG}_DONE"
done

# --- [3] 判決：映射回假色座標系 → 對 GT 算 AUC ------------------------------
mark VERDICT_START
$PY3 - > ~/e26_verdict.txt 2>&1 <<'PYEOF'
import json, os
import numpy as np, pandas as pd

H = os.path.expanduser("~")
meta = json.load(open(f"{H}/prep_meta.json"))
gt = pd.read_csv(f"{H}/2026training.csv")
p = gt["ID"].str.rsplit("_", n=1, expand=True)
gt["seq"], gt["frame"] = p[0], p[1].astype(int)

def iou(a, b):
    ix1 = np.maximum(a[:, 0], b[:, 0]); iy1 = np.maximum(a[:, 1], b[:, 1])
    ix2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2])
    iy2 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    it = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return it / np.maximum(a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - it, 1e-9)

def load(leg):
    d = pd.read_csv(f"{H}/out_{leg}/submission.csv")
    q = d["ID"].str.rsplit("_", n=1, expand=True)
    d["seq"], d["pos"] = q[0], q[1].astype(int)
    out = []
    for s, g in d.groupby("seq"):
        m = meta[s]; x1, y1 = m["win"][0], m["win"][1]
        sc = m["macro"] if leg == "mosaic" else 1        # mosaic 腿需除回 macro
        fids = m["frame_ids"]
        g = g.sort_values("pos").copy()
        g["frame"] = [fids[i - 1] for i in g["pos"]]     # prep 用 1..N 連續編號
        g["x"] = g["x"] / sc + x1; g["y"] = g["y"] / sc + y1
        g["width"] = g["width"] / sc; g["height"] = g["height"] / sc
        out.append(g[["seq", "frame", "x", "y", "width", "height"]])
    return pd.concat(out, ignore_index=True)

rows, pooled = [], {}
for leg in ("fc", "mosaic"):
    pr = load(leg)
    m = gt.merge(pr, on=["seq", "frame"], suffixes=("_g", "_p"))
    m["iou"] = iou(m[["x_g", "y_g", "width_g", "height_g"]].to_numpy(float),
                   m[["x", "y", "width", "height"]].to_numpy(float))
    pooled[leg] = float(m["iou"].mean())
    for s, g in m.groupby("seq"):
        rows.append({"leg": leg, "seq": s, "n": len(g), "auc": float(g["iou"].mean()),
                     "lost_rate": float((g["iou"] < 0.1).mean())})

df = pd.DataFrame(rows).pivot(index="seq", columns="leg", values="auc")
df["delta"] = df["mosaic"] - df["fc"]
print(df.round(4).to_string())
print(f"\npooled: fc {pooled['fc']:.4f} | mosaic {pooled['mosaic']:.4f} "
      f"| Δ {pooled['mosaic']-pooled['fc']:+.4f}")
print(f"健全性錨點：E15 無 crop 三支 pooled = 0.1510 → fc 腿 {'✅ 高於' if pooled['fc'] > 0.1510 else '🚨 未高於'}（crop 本身有效才算實驗成立）")

n_better = int((df["delta"] > 0).sum()); worst = float(df["delta"].min())
print(f"\n事前判準：≥2 支改善 且 最壞 >−0.05")
print(f"  改善 {n_better}/3 支｜最壞 {worst:+.4f}")
if n_better >= 2 and worst > -0.05:
    verdict = "✅ 判準達成 → 擴到全量 test（21 支 crop 選中序列）後發 LB"
elif worst < -0.10 or n_better < 2:
    verdict = "❌ 判準未達 → 結案，記錄逐序列 delta 與型態"
else:
    verdict = "⚠️ 中間帶 → 看型態，需 Sean 裁示是否條件式（D037 教訓：條件式易失敗）"
print(f"▶ {verdict}")
json.dump({"pooled": pooled, "per_seq": rows, "n_better": n_better,
           "worst": worst, "verdict": verdict},
          open(f"{H}/e26_verdict.json", "w"), indent=1, ensure_ascii=False)
PYEOF
cat ~/e26_verdict.txt
rclone copyto ~/e26_verdict.txt "$GOUT/verdict.txt"
rclone copyto ~/e26_verdict.json "$GOUT/verdict.json" 2>/dev/null || true
rclone copyto ~/prep_meta.json "$GOUT/prep_meta.json"
mark VERDICT_DONE
rclone copyto "$STAMP" "$GOUT/timing.txt"

echo "===================== E26 完成 ====================="
cat "$STAMP"
echo "⚠️ 記得 terminate：$INSTANCE_NAME（D018 沒有『已停止』狀態，OS shutdown 仍計費）"
