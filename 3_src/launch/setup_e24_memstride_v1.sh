#!/usr/bin/env bash
# ============================================================================
# setup_e24_memstride_v1.sh — E24：memory_temporal_stride_for_eval r=1 → 5
#
# 為什麼是這條：D046 把主線定為「抑制干擾物」，但三條「把 crop 用到更多序列」的退路
# 全數封閉（D043 門檻放寬 / D043 分段窗 / D045 滑動窗）→ 新槓桿必須來自 crop 以外。
# E18 已證明介入點必須在 **memory 層**（mask 選擇層無效，SAMURAI Kalman ±0.002）。
# 本旋鈕正是 memory 層、且零工程量。
#
# 機制（對得上 E15 診斷的病灶時序；控制流已逐字核對上游原始碼，非推測）：
#   num_maskmem=7 → 1 個 conditioning 槽（首幀 GT）+ 6 個非條件槽。
#   ⚠️ 兩 backend 同名同預設 r=1，但**走不同程式路徑**——事前查清楚才不會重蹈 E18
#      「機制沒生效卻以為在跑」的覆轍：
#   [SAM3] use_memory_selection 預設 True（build_sam3_video_model 的 apply_temporal_disambiguation=True），
#          _prepare_memory_conditioned_features 先算 `r = 1 if self.training else
#          self.memory_temporal_stride_for_eval`，再把 r 傳進 frame_filter 當 `step = -r`：
#          自 t-1 往回以 r 為間隔掃描，收集 eff_iou_score > mf_threshold(0.01) 的幀（上限
#          max_obj_ptrs_in_encoder-1 = 15），6 個 maskmem 槽取 valid_indices[-t_rel] ＝最新 6 個；
#          `must_include = frame_idx-1` 保證最近幀恆在。
#          注意迴圈內的槽位指派走 valid_indices，**stride 公式那條 else 分支被跳過**——
#          r 的作用點是 frame_filter 的掃描步長，不是那條公式。
#   [SAM2] 無 use_memory_selection → 走 else 分支的
#          prev_frame_idx = ((frame_idx-2)//r)*r - (t_rel-2)*r。
#   兩條路徑的淨效果同向：r=1 時 6 槽只覆蓋 t-1..t-6 → identity switch 後約 6 幀整個記憶庫
#   就被錯的目標填滿 ＝「單一時刻跳變後穩住、切換後 IoU 仍高」的成因（1 個 GT 錨點打 21 個近期錯證據）。
#   r=5 使 6 槽覆蓋 t-1,t-6,t-11,t-16,t-21,t-26 ＝跨度 6→26 幀（×4.3）→ 正確身分的證據要 26 幀才被沖乾淨。
#   槽的位置編碼索引不變（t_pos 仍 1..6），只改哪些幀填進槽 → 這是 Meta 設計的 eval 期旋鈕。
#
# 為什麼值得燒這一輪（本地首度有高訊噪比標的，不受 ±0.023 限制）：
#   val 3 支無人機群 pooled：E02(SAM2.1) 0.62147 → E15(SAM3) 0.18367，CLE 5.39 → 90.71px。
#   修好即 val 0.68843 → 0.70499（+0.0166）。訊號量級 +0.44 ≫ 雜訊底線。
#
# 底座無關（D041 等待期方針）：SAM3 與 SAM2 同名同預設（皆 r=1，已逐字核對兩邊原始碼）
#   → 同一招可同時打主線與 Apache-2.0 備案。**本輪只跑 SAM3；SAM2.1 版留待明日視結果。**
# ⚠️ num_maskmem 不可比照辦理：它被烘進 maskmem_tpos_enc 的學習參數形狀，事後改會索引錯位。
#
# 紀律：D036（單 tar ∥ 裝環境）、D016 + 08-07 教訓（**每階段完成即 rclone**，不留到最後）、
#       D018（Lambda 關機仍計費，必須 API terminate）、D014（官方 rclone）、
#       CLAUDE.md（隔離 venv、版本號檔名、不用 ls|head、trap ERR）。
#
# ⏰ 計時器盤點：本腳本**只掛一個** sleep（最後的 4 小時自毀保險）。
#    延長死線前務必 `pgrep -af "sleep [0-9]+"` 確認沒有第二套（cloud-init 的 rearm_selfkill）。
#    4h ≫ 本輪預估 1.5h，不會腰斬實驗（08-07 的 drill.sh 4h 保險砍掉差 3 分鐘完成的實驗）。
# ============================================================================
set -euo pipefail
trap 'echo "🚨 腳本死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
RUN="e24_memstride_20260807"
DEST="$GDRIVE/5_outputs/$RUN"
VAL=~/val_fc
TEST=~/test_fc
CKPT=~/ckpt
PY=~/sam3env/bin/python
export STRIDE="${STRIDE:-5}"                # 文獻預設值（XMem/Cutie），D033 不網格搜索
                                            # export 是必要的：GATE heredoc 的 python 讀 os.environ
INSTANCE_NAME="${INSTANCE_NAME:-hsot-e24-memstride}"
CKPT_FILE=sam3.pt
CKPT_URL="https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt"
[ -n "${HF_TOKEN:-}" ] && CKPT_URL="https://huggingface.co/facebook/sam3/resolve/main/sam3.pt"

# E15（r=1）在 3 支病灶序列上的基準，本機已算好，寫死供閘門比對
BASE_DRONE_POOLED=0.18367

echo "=== E24：memory_temporal_stride_for_eval 1 → $STRIDE | SAM3 860M ==="
mkdir -p "$VAL" "$TEST" "$CKPT"

# --- [0/8] rclone（D014）----------------------------------------------------
command -v rclone >/dev/null 2>&1 || curl -fsSL https://rclone.org/install.sh | sudo bash
rclone version | head -1
rclone lsf "$GDRIVE/" >/dev/null || { echo "🚨 gDrive 存取失敗（rclone.conf 沒推上來？）"; exit 1; }

# --- [1/8] 背景拉資料（與環境安裝並行——D036）-------------------------------
echo "=== [1/8] 背景拉 val + test 單 tar（D036）==="
(
  set -euo pipefail
  if [ ! -f "$VAL/.done" ]; then
    rclone copyto "$GDRIVE/1_data/packed/t1val_fc_65.tar" ~/val.tar
    tar -xf ~/val.tar -C "$VAL"; touch "$VAL/.done"
  fi
  if [ ! -f "$TEST/.done" ]; then
    rclone copyto "$GDRIVE/1_data/packed/t1test_fc_75.tar" ~/test.tar
    tar -xf ~/test.tar -C "$TEST"; touch "$TEST/.done"
  fi
  rclone copyto "$GDRIVE/1_data/val_split_v1.txt"    ~/val_split_v1.txt
  rclone copyto "$GDRIVE/1_data/raw/2026training.csv" ~/2026training.csv
  rclone copyto "$GDRIVE/3_src/track_t1.py"          ~/track_t1.py
  rclone copyto "$GDRIVE/3_src/hsot/eval.py"         ~/eval_hsot.py
  # E15（r=1）val 基準，供逐序列 delta 比對（D033）
  rclone copyto "$GDRIVE/5_outputs/e15_sam3_20260806/sam3/submission.csv" ~/e15_val_base.csv
  echo "DATA_READY"
) > ~/data_pull.log 2>&1 &
DATA_PID=$!

# --- [2/8] 隔離 venv（CLAUDE.md：禁 --system-site-packages）-----------------
echo "=== [2/8] SAM3 環境（完全隔離 venv）==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
[ -d ~/sam3env ] || uv venv --python 3.12 ~/sam3env
VIRTUAL_ENV=~/sam3env uv pip install -q torch torchvision --torch-backend=auto
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'; print('torch', torch.__version__, torch.version.cuda)"

# --- [3/8] sam3 套件 ∥ 權重（四坑已固化，見 setup_e15_sam3_lambda_v2.sh 註解）--
echo "=== [3/8] sam3 套件 + 權重（3.2 GiB）==="
(
  set -euo pipefail
  if [ ! -f "$CKPT/$CKPT_FILE" ]; then
    if [ -n "${HF_TOKEN:-}" ]; then
      curl -fL -H "Authorization: Bearer $HF_TOKEN" -o "$CKPT/$CKPT_FILE" "$CKPT_URL"
    else
      curl -fL -o "$CKPT/$CKPT_FILE" "$CKPT_URL"
    fi
  fi
  echo "CKPT_READY"
) > ~/ckpt_pull.log 2>&1 &
CKPT_PID=$!

VIRTUAL_ENV=~/sam3env uv pip install -q "git+https://github.com/facebookresearch/sam3.git" \
  "numpy<2" ftfy==6.1.1 regex einops psutil scipy av \
  pycocotools numba python-rapidjson pandas pillow tqdm opencv-python-headless
VIRTUAL_ENV=~/sam3env uv pip install -q "setuptools<81"   # 必須最後降級（81+ 移除 pkg_resources）
$PY -c "
import sam3, numpy, setuptools
assert numpy.__version__.startswith('1.'), f'numpy {numpy.__version__} 必須 <2'
assert int(setuptools.__version__.split('.')[0]) < 81, 'setuptools 已移除 pkg_resources'
print(f'sam3 OK | numpy {numpy.__version__} | setuptools {setuptools.__version__}')
" 2>&1 | grep -viE 'userwarning|deprecat|^ +import'

wait $CKPT_PID || { echo "🚨 權重下載失敗（見 ckpt_pull.log）"; exit 1; }
sz=$(stat -c%s "$CKPT/$CKPT_FILE")
[ "$sz" -gt 3000000000 ] || { echo "🚨 權重 ${sz}B 太小（應 ~3.2GiB）"; exit 1; }
echo "ckpt OK ($((sz/1000000)) MB)"

wait $DATA_PID || { echo "🚨 資料拉取失敗（見 data_pull.log）"; exit 1; }
nval=$(find "$VAL"  -mindepth 1 -maxdepth 1 -type d | wc -l)
ntst=$(find "$TEST" -mindepth 1 -maxdepth 1 -type d | wc -l)
nini=$(find "$TEST" -name init_rect.txt | wc -l)
[ "$nval" -ge 65 ] && [ "$ntst" -ge 75 ] && [ "$nini" -ge 75 ] \
  || { echo "🚨 val $nval(應65) / test $ntst(應75) / init_rect $nini(應75)"; exit 1; }
echo "資料就緒：val $nval、test $ntst、init_rect $nini"

# --- [4/8] 閘門一：旋鈕真的存在且會生效（E18 教訓：機制沒生效卻以為在跑）-----
echo "=== [4/8] 閘門一：旋鈕存在性 + 生效驗證 ==="
$PY - "$CKPT/$CKPT_FILE" "$STRIDE" <<'PROBE'
import sys
ckpt, stride = sys.argv[1], int(sys.argv[2])
from sam3.model_builder import build_sam3_video_model
m = build_sam3_video_model(checkpoint_path=ckpt, device="cuda")
t = m.tracker; t.backbone = m.detector.backbone
for n in ["init_state", "add_new_points_or_box", "propagate_in_video"]:
    assert hasattr(t, n), f"❌ tracker 缺 {n}"
A = "memory_temporal_stride_for_eval"
assert hasattr(t, A), f"❌ 無 {A} —— 上游已改名，停工確認"
assert getattr(t, A) == 1, f"❌ 預設不是 1 而是 {getattr(t, A)}——基準假設錯誤，停工重算"
# ★ 控制流驗證（E18 教訓：屬性設得進去 ≠ 執行時會讀它）。
#   SAM3 走 use_memory_selection 分支 → r 的作用點是 frame_filter 的掃描步長；
#   若上游哪天把 apply_temporal_disambiguation 預設改掉，會落到 stride 公式那條，
#   機制仍在但覆蓋跨度算法不同 → 兩條都可接受，但必須知道自己在哪條，否則無法歸因。
ums = getattr(t, "use_memory_selection", None)
assert ums is True, (f"❌ use_memory_selection={ums}（預期 True）——上游預設已變，"
                     f"控制流分支與本實驗的機制推導不符，停工重新核對原始碼")
assert hasattr(t, "frame_filter"), "❌ 無 frame_filter —— r 的作用點不存在，停工確認"
setattr(t, A, stride); assert getattr(t, A) == stride
nm = getattr(t, "num_maskmem", 7)
print(f"✅ 閘門一過：{A} 1 → {stride}"
      f"（走 use_memory_selection 分支：frame_filter step=-{stride}，"
      f"{nm-1} 個 maskmem 槽跨度 {nm-1} → {stride*(nm-1)} 幀）"
      f" | mf_threshold={getattr(t,'mf_threshold','n/a')}"
      f" | max_obj_ptrs={getattr(t,'max_obj_ptrs_in_encoder','n/a')}")
PROBE

# --- [5/8] 閘門二：canary 8 支（3 病灶 + 5 穩定），**判準事前寫死**（D045）----
echo "=== [5/8] 閘門二：canary 3 病灶 + 5 穩定 ==="
printf 'vis-droneshow2\nrednir-droneshow2\nrednir-drone2\n' > ~/drone3.txt
printf 'vis-officefan2\nnir-redbag\nrednir-glass2\nvis-receipts3\nnir-glass_cup\n' > ~/stable5.txt
cat ~/drone3.txt ~/stable5.txt > ~/canary8.txt
$PY ~/track_t1.py --frames-root "$VAL" --seq-list ~/canary8.txt \
  --gt-csv ~/2026training.csv --out-dir ~/out_canary \
  --backend sam3 --sam3-ckpt "$CKPT/$CKPT_FILE" --memory-stride "$STRIDE"
rclone copy ~/out_canary "$DEST/canary" --transfers 8      # 08-07 教訓：先回傳再評估

$PY - "$BASE_DRONE_POOLED" <<'GATE'
import sys, json
sys.path.insert(0, "/home/ubuntu")
from eval_hsot import evaluate
base_pooled = float(sys.argv[1])
GT, PRED = "/home/ubuntu/2026training.csv", "/home/ubuntu/out_canary/submission.csv"
E15 = {"vis-droneshow2": 0.1009, "rednir-droneshow2": 0.1349, "rednir-drone2": 0.3415,
       "vis-officefan2": 0.9594, "nir-redbag": 0.9494, "rednir-glass2": 0.9114,
       "vis-receipts3": 0.9279, "nir-glass_cup": 0.8913}
drone = [l.strip() for l in open("/home/ubuntu/drone3.txt") if l.strip()]
stable = [l.strip() for l in open("/home/ubuntu/stable5.txt") if l.strip()]
r = evaluate(PRED, GT, seqs=drone)
g = lambda v: v["auc"] if isinstance(v, dict) else v
# ⚠️ evaluate() 的 'pooled' 也是 dict（auc/dp20/cle/n_frames），與 per_seq 同構——
#    初版漏了這一層，直接 f"{pooled:.5f}" 炸 TypeError: unsupported format string
#    passed to dict.__format__，害整個閘門死掉（追蹤本身沒事，因為 rclone 在評估之前）。
pooled = g(r["pooled"])
cle = r["pooled"]["cle"] if isinstance(r["pooled"], dict) else float("nan")
per = evaluate(PRED, GT, seqs=drone + stable)["per_seq"]

print(f"\n{'序列':32s} {'E15(r=1)':>10s} {'E24(r=%d)'%int(__import__('os').environ.get('STRIDE',5)):>10s} {'Δ':>9s}")
worst_stable = 0.0
for s in drone + stable:
    new, old = g(per[s]), E15[s]
    d = new - old
    if s in stable:
        worst_stable = min(worst_stable, d)
    print(f"{s:32s} {old:10.4f} {new:10.4f} {d:+9.4f}   {'← 病灶' if s in drone else ''}")
print(f"\n3 支病灶 pooled：{base_pooled:.5f} → {pooled:.5f}（Δ {pooled-base_pooled:+.5f}）")
print(f"CLE：90.71 → {cle:.2f}px")
print(f"穩定組最壞 Δ：{worst_stable:+.4f}")

# ---- 事前寫死的判準（D045：一發資源應買到「決策」而非「一個數字」）----
if worst_stable < -0.05:
    verdict, action = "災難", "穩定組崩壞 → 停止，不投 LB 額度"
elif pooled > 0.30:
    verdict, action = "機制成立", "病灶被觸動 → 續跑全量，明天第 1 發"
elif pooled > 0.22:
    verdict, action = "部分有效", "方向對但量級小 → 續跑全量，明天第 2–3 發"
else:
    verdict, action = "機制不成立", "病灶未被觸動（同 E18 型態）→ 仍跑 test 當全域探針，降至第 4 發"
print(f"\n判決：【{verdict}】{action}")
json.dump({"verdict": verdict, "action": action, "drone_pooled": pooled,
           "drone_delta": pooled - base_pooled, "worst_stable": worst_stable,
           "per_seq": {s: g(per[s]) for s in drone + stable}},
          open("/home/ubuntu/out_canary/gate_verdict.json", "w"), indent=1, ensure_ascii=False)
sys.exit(2 if verdict == "災難" else 0)
GATE
rclone copyto ~/out_canary/gate_verdict.json "$DEST/canary/gate_verdict.json"

# --- [6/8] 全量 val 65（斷點續跑；canary 8 支已在 out_val 外，需重跑）--------
echo "=== [6/8] 全量 val 65（r=$STRIDE）==="
$PY ~/track_t1.py --frames-root "$VAL" --seq-list ~/val_split_v1.txt \
  --gt-csv ~/2026training.csv --out-dir ~/out_val \
  --backend sam3 --sam3-ckpt "$CKPT/$CKPT_FILE" --memory-stride "$STRIDE"
rclone copy ~/out_val "$DEST/val65" --transfers 8          # 立即回傳
echo "✅ val 65 已回傳 $DEST/val65/"

$PY - <<'VALEVAL'
import sys; sys.path.insert(0, "/home/ubuntu")
from eval_hsot import evaluate
seqs = [l.strip() for l in open("/home/ubuntu/val_split_v1.txt") if l.strip()]
new = evaluate("/home/ubuntu/out_val/submission.csv", "/home/ubuntu/2026training.csv", seqs=seqs)
old = evaluate("/home/ubuntu/e15_val_base.csv",       "/home/ubuntu/2026training.csv", seqs=seqs)
print(f"\nval 65 pooled：E15(r=1) {old['pooled']:.5f} → E24 {new['pooled']:.5f}"
      f"（Δ {new['pooled']-old['pooled']:+.5f}）")
print("⚠️ 本數字的 bootstrap SE = ±0.023（D040）→ 只當災難檢查，不當增益閘門。"
      "真正的訊號在上面 canary 的 3 支病灶。")
g = lambda v: v["auc"] if isinstance(v, dict) else v
d = sorted(((g(new['per_seq'][s]) - g(old['per_seq'][s]), s) for s in seqs if s in new['per_seq'] and s in old['per_seq']))
print("\n最壞 5 支：", "  ".join(f"{s} {x:+.3f}" for x, s in d[:5]))
print("最佳 5 支：", "  ".join(f"{s} {x:+.3f}" for x, s in d[-5:][::-1]))
VALEVAL

# --- [7/8] test 75（明日 LB 探針的材料）-------------------------------------
echo "=== [7/8] test 75（r=$STRIDE）==="
$PY ~/track_t1.py --frames-root "$TEST" --out-dir ~/out_test \
  --backend sam3 --sam3-ckpt "$CKPT/$CKPT_FILE" --memory-stride "$STRIDE"
rclone copy ~/out_test "$DEST/test75" --transfers 8        # 立即回傳
echo "✅ test 75 已回傳 $DEST/test75/"

# --- [8/8] 收尾 -------------------------------------------------------------
echo "=== [8/8] 完成摘要 ==="
$PY -c "
import json
v=json.load(open('/home/ubuntu/out_canary/gate_verdict.json'))
print('閘門判決：', v['verdict'], '|', v['action'])
print('3 支病灶 pooled Δ：', round(v['drone_delta'], 5))
"
echo "產出：$DEST/{canary,val65,test75}/"
echo "下一步（本機）：hsot/compare_runs.py 對 E15 基準看逐序列 delta；決定明日發序"

# --- 自毀保險（D018；本腳本唯一的 sleep）------------------------------------
nohup bash -c '
  sleep 2400
  key=$(cat ~/.lambda_key | tr -d "\n"); [ -z "$key" ] && exit 0
  id=$(curl -s -u "$key:" https://cloud.lambdalabs.com/api/v1/instances | python3 -c "
import json,sys,os
d=json.load(sys.stdin).get(\"data\",[])
m=[i[\"id\"] for i in d if i.get(\"name\")==os.environ.get(\"INSTANCE_NAME\",\"hsot-e24-memstride\")]
print(m[0] if m else \"\")")
  [ -n "$id" ] && curl -s -u "$key:" -X POST \
    https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
    -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$id\"]}"
' > ~/self_destruct.log 2>&1 &
echo "⏰ 4 小時自毀保險已啟動（本輪預估 1.5h，不會腰斬）。驗收完請提前手動 terminate 省錢。"
echo "   盤點所有計時器：pgrep -af 'sleep [0-9]+'"
