#!/usr/bin/env bash
# ============================================================================
# setup_e23_sam21_cropzoom_v1.sh — E23：SAM2.1-L + crop-zoom（D041 授權備案 + 底座消融）
#
# 為什麼要跑（D041）：SAM 3 的 LICENSE 是 Meta 自訂「SAM License」，非 OSI 核准且含用途
# 限制，字面上牴觸 Kaggle Foundational Rules §6c。§9b(ii) 雖給一週補救期，但一週不足以
# 從零重建備案 → **備案必須是現成的**。SAM 2/2.1 是 Apache 2.0，完全合規。
#
# 產出兩份（同一輪，crop prep 是 CPU、追蹤才是 GPU 成本，各約 9 分鐘）：
#
#   A) e23a_fallback.csv —— **真正的 SAM3-free 備案**
#      窗 = E02 ∪ E03 軌跡聯集（SAMURAI+SAM2.1-L 與 DAM4SAM+SAM2.1-L，兩條**獨立的
#      SAM2.1 系**軌跡）。**關鍵設計**：若 SAM3 被禁，則用 SAM3 軌跡算出來的裁切窗
#      同樣不能用 → 窗源必須也是 SAM3-free。同時保留 E19 的聯集窗結構性修正
#      （單一軌跡跟丟時凍結在錯位置，其 envelope 特別小、會通過面積檢查而把真目標切在窗外）。
#      → 回答：「若 SAM3 不能用，我方還剩多少分？」
#
#   B) e23b_ablation.csv —— **純底座消融**
#      窗 = E02 ∪ E15（與 v008/E19 逐位元相同），只把 tracker 由 SAM3 換成 SAM2.1-L。
#      → 回答：「0.69128 之中，底座(SAM3) 與前處理(crop-zoom) 各值多少？」
#      注意：B 含 SAM3 衍生資訊，**不可作為備案**，只作分析用。
#
# 兩者的 merge base 都是 E02（0.66608），故與 v008 的差即為完整替換效果。
#
# 為何 E14 的 −0.001 不算數：落在 val ±0.023 雜訊帶內＝從未真正被量測（D040），
# 且未含 E19 才修掉的聯集窗 bug、窗參數也不同 → SAM2.1+聯集窗 crop-zoom 是空白。
# 偏悲觀的先驗：E19 診斷顯示小目標凍結率 E02 0.083 < SAM3 0.110，即 SAM2.1 在小目標上
# 本就沒那麼糟，crop-zoom 可修空間可能較小。
#
# 環境：**沿用產出 E02 的原始配方**（venv --system-site-packages 繼承 Lambda Stack 系統
# torch + yangchris11/samurai）。刻意不改成隔離 venv——改了就無法保證與 E02 可比。
#
# 紀律：D036（單 tar）、D018（用完必 API terminate）、D016（產出漸進回傳）、D014（官方 rclone）。
# 成功不自動自毀（v1 的假成功差點銷毀資產）——由本機確認產出後主動 terminate。
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
PY=~/t1env/bin/python
SAMURAI=~/samurai
OUT=~/e23; mkdir -p "$OUT"
INSTANCE_NAME="${INSTANCE_NAME:-hsot-e23}"
die() { echo "🚨 $*"; exit 1; }

# --- 保險自毀（開頭就掛：D018，Lambda 關機仍計費）----------------------------
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
echo "⏰ 4 小時保險自毀已掛"

echo "=== [0] rclone ==="
command -v rclone >/dev/null || curl -fsSL https://rclone.org/install.sh | sudo bash
rclone lsf "$GDRIVE/" >/dev/null || die "gDrive 失敗（rclone.conf 沒推上來？）"

echo "=== [1] 背景拉 test 假色單 tar（D036）==="
(
  set -euo pipefail
  mkdir -p ~/test_fc
  rclone copyto "$GDRIVE/1_data/packed/t1test_fc_75.tar" ~/t1test.tar
  tar -xf ~/t1test.tar -C ~/test_fc
  d=$(find ~/test_fc -mindepth 1 -maxdepth 1 -type d | wc -l)
  n=$(find ~/test_fc -name init_rect.txt | wc -l)
  [ "$d" -eq 75 ] && [ "$n" -eq 75 ] || { echo "🚨 序列 $d / init_rect $n（應各 75）"; exit 1; }
  echo "TEST_FC_READY"
) > ~/fc.log 2>&1 &
FC_PID=$!

echo "=== [2] t1env（E02 原始配方：繼承系統 torch）+ SAMURAI ==="
if [ ! -d ~/t1env ]; then
  python3 -m venv --system-site-packages ~/t1env
fi
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'; print('torch', torch.__version__)"
[ -d "$SAMURAI" ] || git clone --depth 1 https://github.com/yangchris11/samurai.git "$SAMURAI"
$PY -m pip install -q -e "$SAMURAI/sam2" loguru tqdm 2>&1 | tail -2
$PY -c "import sam2; print('sam2 import OK')"

echo "=== [3] SAM2.1-Large 權重 ==="
mkdir -p ~/ckpt
[ -f ~/ckpt/sam2.1_hiera_large.pt ] || rclone copy "$GDRIVE/4_models/pretrained/sam2.1_hiera_large.pt" ~/ckpt/
sz=$(stat -c%s ~/ckpt/sam2.1_hiera_large.pt)
[ "$sz" -gt 800000000 ] || die "權重 ${sz}B 太小，非 Large"
echo "ckpt OK ($((sz/1000000)) MB)"

echo "=== [4] 程式碼與三條軌跡 CSV ==="
rclone copyto "$GDRIVE/3_src/track_t1.py" ~/track_t1.py
rclone copy "$GDRIVE/3_src/hsot/" ~/hsot/ --include "*.py"
rclone copyto "$GDRIVE/5_outputs/submissions/exp003_samurai_large.csv" ~/e02.csv   # SAMURAI+SAM2.1-L
rclone copyto "$GDRIVE/5_outputs/submissions/exp004_dam4sam_large.csv" ~/e03.csv   # DAM4SAM+SAM2.1-L
rclone copyto "$GDRIVE/5_outputs/submissions/sub_v006_e15sam3.csv"     ~/e15.csv   # SAM3（僅供 B 消融）
for f in ~/e02.csv ~/e03.csv ~/e15.csv; do
  [ -s "$f" ] || die "$f 缺失或為空"
done

wait $FC_PID; grep -q TEST_FC_READY ~/fc.log || die "test 假色失敗（見 ~/fc.log）"
echo "✅ 環境 + 權重 + 資料 + 程式碼就緒"

# --- 閘門一：SAM2.1 backend 冒煙（三模態各一，未裁切原圖）-------------------
echo "=== [5] 閘門一：SAMURAI backend 冒煙 3 支 ==="
SMOKE=~/smoke_e23.txt; : > "$SMOKE"
for mod in vis nir rednir; do
  find ~/test_fc -mindepth 1 -maxdepth 1 -type d -name "${mod}-*" -print -quit \
    | xargs -r -n1 basename >> "$SMOKE"
done
cat "$SMOKE"
rm -rf ~/out_smoke
$PY ~/track_t1.py --frames-root ~/test_fc --seq-list "$SMOKE" --out-dir ~/out_smoke \
  --backend samurai --samurai-dir "$SAMURAI" --ckpt ~/ckpt/sam2.1_hiera_large.pt \
  || die "冒煙執行失敗"
$PY - <<'CHK' || die "冒煙驗收失敗"
import json, os, pandas as pd, numpy as np
H = os.path.expanduser
d = json.load(open(H("~/out_smoke/diagnostics.json")))
bad = [k for k, v in d.items() if k != "_meta" and "error" in v]
assert not bad, f"❌ 冒煙失敗 {bad}"
assert d["_meta"]["backend"] == "samurai", f"backend 記為 {d['_meta']['backend']}"
# 關鍵驗證：未裁切原圖 + SAM2.1-L 應「逐幀重現 E02」（E02 就是這個組態產生的）。
# 若對不上，代表環境/權重/設定與 E02 不同，後續與 E02 的比較就不成立。
new = pd.read_csv(H("~/out_smoke/submission.csv"))
ref = pd.read_csv(H("~/e02.csv"))
m = ref["ID"].isin(new["ID"])
a = new.set_index("ID").loc[ref.loc[m, "ID"]][["x", "y", "width", "height"]].to_numpy()
b = ref.loc[m, ["x", "y", "width", "height"]].to_numpy()
diff = (a != b).any(1)
mx = float(np.abs(a - b).max()) if len(a) else 0.0
print(f"vs E02 逐幀比對：{len(a)} 幀，不同 {diff.sum()} 幀（{diff.mean():.2%}），最大座標差 {mx:.1f}px")
assert diff.mean() < 0.05, f"❌ 與 E02 差異 {diff.mean():.1%} 過大 —— 環境/設定與 E02 不一致，比較不成立"
fps = [v["fps"] for k, v in d.items() if k != "_meta" and "fps" in v]
print(f"✅ 閘門一過：3/3 成功，FPS {fps}，與 E02 一致")
CHK

# --- crop prep ×2（CPU，便宜）------------------------------------------------
# A：SAM3-free（窗 = E02 ∪ E03）—— 真正的備案
echo "=== [6a] crop prep A：窗 = E02 ∪ E03（SAM3-free）==="
PYTHONPATH=~ $PY -m hsot.crop_rerun prep --frames-root ~/test_fc \
  --base-csv ~/e02.csv --envelope-extra ~/e03.csv \
  --out-root ~/crop_a --meta ~/meta_a.json | tail -4
NA=$($PY -c "import json;print(len(json.load(open('$HOME/meta_a.json'))))")
[ "$NA" -ge 5 ] || die "A 只選中 $NA 支，不值得跑"
$PY -c "
import json,os
m=json.load(open(os.path.expanduser('~/meta_a.json')))
open(os.path.expanduser('~/seqs_a.txt'),'w').write('\n'.join(sorted(m))+'\n')"

# B：與 E19 同窗（E02 ∪ E15）—— 純底座消融，含 SAM3 衍生資訊，不可作備案
echo "=== [6b] crop prep B：窗 = E02 ∪ E15（與 v008/E19 相同）==="
PYTHONPATH=~ $PY -m hsot.crop_rerun prep --frames-root ~/test_fc \
  --base-csv ~/e15.csv --envelope-extra ~/e02.csv \
  --out-root ~/crop_b --meta ~/meta_b.json | tail -4
NB=$($PY -c "import json;print(len(json.load(open('$HOME/meta_b.json'))))")
[ "$NB" -eq 21 ] || echo "⚠️ B 選中 $NB 支（E19 當時 21 支）——規則應確定性，請查明"
$PY -c "
import json,os
m=json.load(open(os.path.expanduser('~/meta_b.json')))
open(os.path.expanduser('~/seqs_b.txt'),'w').write('\n'.join(sorted(m))+'\n')"
echo "選中：A=$NA 支（SAM3-free）｜B=$NB 支（同 E19 窗）"

# --- 追蹤 ×2 + merge（merge base 皆為 E02）----------------------------------
names_a="e23a_fallback"; names_b="e23b_ablation"
for V in a b; do
  eval "OUTNAME=\$names_$V"
  echo "=== [7$V] SAM2.1-L 追蹤裁切序列（變體 $V → $OUTNAME）==="
  $PY ~/track_t1.py --frames-root ~/crop_$V --seq-list ~/seqs_$V.txt \
    --out-dir "$OUT/crop_$V" --backend samurai --samurai-dir "$SAMURAI" \
    --ckpt ~/ckpt/sam2.1_hiera_large.pt || die "變體 $V 追蹤失敗"
  PYTHONPATH=~ $PY -m hsot.crop_rerun merge --base-csv ~/e02.csv \
    --crop-csv "$OUT/crop_$V/submission.csv" --meta ~/meta_$V.json \
    --out "$OUT/${OUTNAME}.csv" || die "變體 $V merge 失敗"
  rclone copy "$OUT" "$GDRIVE/5_outputs/e23_sam21_20260806" --transfers 8   # D016 漸進回傳
done

# --- 災難檢查（D040：無 GT，只驗完整性 + 與已知錨點的偏離量級）--------------
$PY - <<'CHK' || die "災難檢查失敗"
import os, pandas as pd, numpy as np
H = os.path.expanduser
e02 = pd.read_csv(H("~/e02.csv"))          # 已知 LB 0.66608，兩變體的 merge base
for name in ("e23a_fallback", "e23b_ablation"):
    df = pd.read_csv(H(f"~/e23/{name}.csv"))
    assert len(df) == len(e02) == 26860, f"{name} 列數 {len(df)}"
    assert (df.ID.values == e02.ID.values).all(), f"{name} ID 順序不符"
    assert df.isna().sum().sum() == 0, f"{name} 有 NaN"
    assert (df.width > 0).all() and (df.height > 0).all(), f"{name} 有非正 w/h"
    ch = (df[["x","y","width","height"]].to_numpy() != e02[["x","y","width","height"]].to_numpy()).any(1)
    d = df.copy(); d["seq"] = d.ID.str.rsplit("_", n=1).str[0]
    cd = np.hypot((df.x + df.width/2) - (e02.x + e02.width/2),
                  (df.y + df.height/2) - (e02.y + e02.height/2))
    per = pd.DataFrame({"seq": d.seq, "cd": cd}).groupby("seq").cd.median().sort_values()
    print(f"[{name}] 變動 {ch.sum():,} 幀 ({ch.mean():.1%})｜vs E02 中心位移中位 {np.median(cd):.1f}px")
    print(f"  偏離最大 5 支：{per.tail(5).round(1).to_dict()}")
CHK
rclone copy "$OUT" "$GDRIVE/5_outputs/e23_sam21_20260806" --transfers 8
rclone copyto ~/meta_a.json "$GDRIVE/5_outputs/e23_sam21_20260806/meta_a.json"
rclone copyto ~/meta_b.json "$GDRIVE/5_outputs/e23_sam21_20260806/meta_b.json"
[ -s "$OUT/e23a_fallback.csv" ] && [ -s "$OUT/e23b_ablation.csv" ] || die "產出不完整，不宣告成功"
echo "E23_DONE"
echo "⚠️ 機器保留；確認 gDrive 產出後由本機主動 terminate（4 小時保險自毀仍在）"
