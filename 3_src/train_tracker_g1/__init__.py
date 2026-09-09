# G1 tracker-only 訓練包（D090 G0 調查 → 策略 (b)）。
# 只新增檔案；不修改任何既有交付物。訓練對象＝SAM3 `tracker.*`（11.7M），
# vision backbone 凍結（perflib fused kernel 在 grad enabled 下會 raise，D088）。
