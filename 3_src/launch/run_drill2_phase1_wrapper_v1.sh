#!/usr/bin/env bash
# run_drill2_phase1_wrapper_v1 — 箱端接力：演練#2 → E-F Phase 1 → 自我 terminate（D018）
# 箱端自主是唯一可靠的安全層（lambda-gpu 鐵律）：不依賴本機 session 存活。
export INSTANCE_NAME=hsot-drill2 PATH="$HOME/.local/bin:$PATH"
GD=gdrive:WHISPERS_2026_HyperSOT

bash ~/setup_coldstart_drill_v2.sh > ~/drill2_console.log 2>&1
DRILL_RC=$?
rclone copyto ~/drill2_console.log "$GD/5_outputs/coldstart_drill2_20260811/drill2_console.log" 2>/dev/null

if [ "$DRILL_RC" -eq 0 ]; then
  bash ~/setup_ef_phase1_v1.sh > ~/phase1_console.log 2>&1
  PH_RC=$?
else
  echo "drill 失敗 rc=$DRILL_RC，跳過 phase1（fail-closed）" > ~/phase1_console.log
  PH_RC=97
fi
echo "WRAPPER_DONE drill_rc=$DRILL_RC phase1_rc=$PH_RC" >> ~/phase1_console.log
rclone copyto ~/phase1_console.log "$GD/5_outputs/ef_phase1_20260811/phase1_console.log" 2>/dev/null

# finish handler：自我 terminate（不留機器過夜；保險計時器與 cloud-init 為兜底）
key=$(tr -d '\n' < ~/.lambda_key 2>/dev/null); [ -z "$key" ] && exit 0
id=$(curl -s -u "$key:" https://cloud.lambda.ai/api/v1/instances | python3 -c "
import json, sys
d = json.load(sys.stdin).get('data', [])
m = [i['id'] for i in d if i.get('name') == 'hsot-drill2']
print(m[0] if m else '')")
[ -n "$id" ] && curl -s -u "$key:" -X POST \
  https://cloud.lambda.ai/api/v1/instance-operations/terminate \
  -H 'Content-Type: application/json' -d "{\"instance_ids\":[\"$id\"]}"
