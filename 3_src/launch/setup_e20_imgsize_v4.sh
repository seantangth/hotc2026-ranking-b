#!/usr/bin/env bash
# ============================================================================
# setup_e20_imgsize_v4.sh — E20：SAM3 image_size 1008→1344/1680（D044 升為 P1 主線）
#
# 【為什麼從「暫緩」翻案】08-07 的 v012/E23b 補完了 2×2 因子表的第四格：
#   crop-zoom 在 SAM2.1 上只值 +0.0059，在 SAM3 上值 +0.0175 —— **差 3 倍**；
#   可加模型預測 v008＝0.67974，實測 0.69128 → **交互作用 +0.0115 比兩個主效應都大**。
#   ⇒ 更強的表徵需要足夠的 patch 預算才發揮得出來。crop-zoom 是**局部**提解析度
#   （只作用 26% 幀、線性 2.05x）就換到 +0.0175；image_size↑ 是**全域**版
#   （100% 幀、線性 1.33–1.67x），且是唯一完全不破壞時間連續性的手段（D043 那一類）。
#
# 【五關全部解開（第 4 關是 08-07 新解，遠比預期簡單）】
#   1–3 RoPE global attention 重建／mask decoder 尺寸／PromptEncoder 三屬性 → 已在 track_t1.py
#   4  **memory encoder**：`SimpleMaskDownSampler.interpol_size` 建構期寫死 [1152,1152]，
#      經 4 層 stride-2 conv 降 16 倍 ＝ 72，正好等於預設的 1008/14。它是**單一真相源**
#      （上游 sam3_video_base 直接讀它決定插值目標），故改成 new_grid*16 即整條路徑一致。
#      exp009 卡住的 `x + masks`（120 vs 72）由此消解 —— 一行屬性，不是動手術。
#   5  position encoding 的 precompute_resolution=1008 **不需動**：只是預填 cache，
#      forward 遇未快取尺寸會現算。
#
# 【硬約束】候選值須**雙重整除**：整除 stride 14 **且** 新網格整除 window 24。
#   1344（網格 96＝24×4）、1680（網格 120＝24×5）合法；**1568 是陷阱**（網格 112 非 24 倍數）。
#
# 【階段】SMOKE（3 支 × 兩個解析度，量崩壞/速度/VRAM）→ 人工判讀 → FULL（全 75 支）
#   閘門只驗「**沒崩**」不驗「有進步」——D040：本地無 GT 且解析度不足以判 <0.05 的效果，
#   真正的裁決在 LB。崩壞（>0.05 級）則是測得出來的。
#
# 用法：bash setup_e20_imgsize_v4.sh smoke      # 先跑這個
#       bash setup_e20_imgsize_v4.sh full 1344  # 判讀後再跑
# 前提：機器上已有 ~/sam3env、~/ckpt/sam3.pt、~/test_fc（冷啟動演練留下的）
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
PY=~/sam3env/bin/python
OUT=~/e20; mkdir -p "$OUT"
MODE="${1:-smoke}"
die() { echo "🚨 $*"; exit 1; }

[ -f ~/ckpt/sam3.pt ] || die "缺 SAM3 權重"
[ -d ~/test_fc ] || die "缺 test 假色資料"
[ -f ~/track_t1.py ] || die "缺 track_t1.py"

if [ "$MODE" = "smoke" ]; then
  # 三模態各一支，固定挑法（不認序列名：取字母序第一支）
  SMOKE=~/smoke_e20.txt; : > "$SMOKE"
  for mod in vis nir rednir; do
    find ~/test_fc -mindepth 1 -maxdepth 1 -type d -name "${mod}-*" -print -quit \
      | xargs -r -n1 basename >> "$SMOKE"
  done
  echo "冒煙序列："; cat "$SMOKE"

  # 基線：1008（＝E15 的組態），同機同環境重跑，作為「同條件對照」
  for SZ in 0 1344 1680; do
    tag=$([ "$SZ" = 0 ] && echo 1008 || echo "$SZ")
    echo "=== 冒煙 image_size=$tag ==="
    rm -rf "$OUT/smoke_$tag"
    /usr/bin/time -v $PY ~/track_t1.py --frames-root ~/test_fc --seq-list "$SMOKE" \
      --out-dir "$OUT/smoke_$tag" --backend sam3 --sam3-version sam3 \
      --sam3-ckpt ~/ckpt/sam3.pt --sam3-image-size "$SZ" \
      > "$OUT/smoke_$tag.log" 2>&1 || { echo "❌ image_size=$tag 執行失敗"; tail -25 "$OUT/smoke_$tag.log"; continue; }
    grep -oE "Maximum resident set size.*|image_size .*" "$OUT/smoke_$tag.log" | head -3
    $PY -c "
import json
d=json.load(open('$OUT/smoke_$tag/diagnostics.json'))
ok=[(k,v) for k,v in d.items() if k!='_meta' and 'error' not in v]
bad=[k for k,v in d.items() if k!='_meta' and 'error' in v]
print(f'  {len(ok)}/{len(ok)+len(bad)} 成功'+(f' ❌ 失敗 {bad}' if bad else ''))
for k,v in ok: print(f'    {k}: {v[\"fps\"]:.1f} fps, empty={v.get(\"empty_ratio\",\"?\")}')"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
  done

  echo ""
  echo "=== 崩壞檢查：各解析度 vs 1008 基線的中心位移 ==="
  $PY - <<'CHK'
import os, itertools
import numpy as np, pandas as pd
H = os.path.expanduser
def load(sz):
    p = H(f"~/e20/smoke_{sz}/submission.csv")
    if not os.path.exists(p): return None
    d = pd.read_csv(p); d["seq"] = d.ID.str.rsplit("_", n=1).str[0]
    return d
base = load(1008)
if base is None:
    print("❌ 1008 基線缺失，無法比較"); raise SystemExit(1)
for sz in (1344, 1680):
    d = load(sz)
    if d is None:
        print(f"  {sz}: 無輸出（執行失敗）"); continue
    m = d.merge(base, on="ID", suffixes=("", "_b"))
    cd = np.hypot((m.x+m.width/2)-(m.x_b+m.width_b/2), (m.y+m.height/2)-(m.y_b+m.height_b/2))
    per = pd.DataFrame({"seq": m.seq, "cd": cd}).groupby("seq").cd.median()
    # 判讀：位移小 = 表徵沒崩（框微調屬正常）；位移巨大 = 追到別的東西了
    verdict = "✅ 未崩壞" if per.max() < 30 else ("⚠️ 有序列大幅偏離" if per.max() < 100 else "❌ 疑似表徵崩壞")
    print(f"  {sz}: 中心位移中位 {np.median(cd):5.1f}px｜逐序列 {per.round(1).to_dict()}  {verdict}")
CHK
  echo ""
  echo "判讀指引：位移中位 <30px＝框微調（正常，解析度提高本就會改變框）；"
  echo "         >100px＝追到別的目標＝位置編碼在新解析度下退化 → 該尺寸判死。"
  echo "接著跑：bash setup_e20_imgsize_v4.sh full <尺寸>"
  exit 0
fi

# ---------------- FULL ----------------
SZ="${2:?full 模式需指定 image_size，例：bash $0 full 1344}"
echo "=== FULL：全 75 支 @ image_size=$SZ ==="
$PY ~/track_t1.py --frames-root ~/test_fc --out-dir "$OUT/full_$SZ" \
  --backend sam3 --sam3-version sam3 --sam3-ckpt ~/ckpt/sam3.pt --sam3-image-size "$SZ" \
  || die "FULL 執行失敗"
cp "$OUT/full_$SZ/submission.csv" "$OUT/e20_${SZ}_full.csv"

# 災難檢查（D040：無 GT，只驗完整性 + 與已知錨點的偏離量級）
$PY - <<CHK || die "災難檢查失敗"
import os
import numpy as np, pandas as pd
H = os.path.expanduser
new = pd.read_csv(H("~/e20/e20_${SZ}_full.csv"))
ref = pd.read_csv(H("~/ref/e15_ref.csv")) if os.path.exists(H("~/ref/e15_ref.csv")) \
      else pd.read_csv(H("~/fresh/e15_fresh.csv"))
assert len(new) == len(ref) == 26860, f"列數 {len(new)}"
assert (new.ID.values == ref.ID.values).all(), "ID 順序不符"
assert new.isna().sum().sum() == 0, "有 NaN"
assert (new.width > 0).all() and (new.height > 0).all(), "有非正 w/h"
m = new.merge(ref, on="ID", suffixes=("", "_r"))
m["seq"] = m.ID.str.rsplit("_", n=1).str[0]
cd = np.hypot((m.x+m.width/2)-(m.x_r+m.width_r/2), (m.y+m.height/2)-(m.y_r+m.height_r/2))
per = pd.DataFrame({"seq": m.seq, "cd": cd}).groupby("seq").cd.median().sort_values()
print(f"✅ 完整性通過｜vs E15(1008) 中心位移中位 {np.median(cd):.1f}px")
print(f"  偏離最大 8 支：{per.tail(8).round(1).to_dict()}")
print(f"  偏離 >100px 的序列數：{int((per>100).sum())}/75（多代表大範圍改變身分，需人工判讀）")
CHK
rclone copy "$OUT" "$GDRIVE/5_outputs/e20_imgsize_20260807" --transfers 8 --include "*.csv" --include "*.json"
echo "E20_FULL_DONE size=$SZ"
