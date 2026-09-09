# 環境版本鎖（D097，2026-08-30）

## 為什麼需要這個目錄

08-30 實測：**這條交付鏈在相同套件版本下是位元級確定性的**
（兩台不同機器、不同日期，產出 md5 完全相同）。
但交付腳本用的是**不帶版本號**的 `uv pip install`，於是：

| 環境 | 同一條 full-SAM3 腿的輸出 | 端到端 LB |
|---|---|---|
| 08-22 建的 | （基準） | — |
| 08-29/30 建的 | 第 12 行即分歧（⚠️ 但這是與 08-22 **不同 profile 的 run** 相比，非單變因） | v083 ＝ **0.70701** |

🚨 **08-30 17:00 更新（run F）：上表的因果解讀已被否證。**
把兩個 venv 降回 08-22 版本重跑（run F），輸出與降版前 **26,860 列全部相同**
⇒ **這些套件版本差異對數值毫無影響**，`v083 = 0.70701` 的落差另有原因（尚未定位）。
**⇒ 本目錄的 lock 檔仍然要用，但理由是工程衛生與可重建性，不是「不釘就會掉 0.01」。**
確定性反而更強：3 次執行 × 2 台機器 × **2 組套件版本**全部位元級相同。

以下保留原始觀察（08-22 vs 08-29/30 的版本差異清單）供溯源：

| venv | 08-22 → 08-29/30 的實際差異 |
|---|---|
| `sam3env` | **timm 1.0.28→1.0.29**、**cuda-pathfinder 1.6.1→1.8.0**、click 8.4.2→8.5.0、filelock、huggingface-hub 1.28→1.29、portalocker、wcwidth |
| `t1env` | **hydra-core 1.3.5→1.3.6**、cuda-pathfinder、filelock、portalocker |

`opencv` 與 `numpy` **兩邊相同**——08-30 中午一度宣稱的「opencv 4.11→5.0 改變 JPEG 解碼」
是 `grep` 跨兩個 freeze 檔取第一個匹配造成的假象，**已作廢**。
**制度教訓：比對版本一律逐檔 `diff`，不要 grep 跨檔。**

## 檔案

| 檔案 | 內容 |
|---|---|
| `sam3env_20260822.lock.txt` | 08-22 run（`rankb_robust_test75_20260822`）的 sam3env `pip freeze`，含 `sam3 @ git+…@96914d24` |
| `t1env_20260822.lock.txt` | 同上的 t1env，**已移除** `-e file:///home/ubuntu/samurai/sam2`（該套件由腳本以 `-e` 就地安裝，路徑因機器而異） |
| `*_20260829_drift.txt` | 08-29 演練 #4 的 freeze，**保留作為「漂移長什麼樣」的對照**，不是要用的版本 |

## 使用方式（封板前需在腳本落實）

兩個 venv 各自：先照現行流程裝 torch（`--torch-backend=auto`）、`sam2`（`-e`）、`sam3`（git SHA），
**再**以對應 lock 檔覆蓋所有可解析套件的版本，最後 assert 關鍵版本：

```bash
VIRTUAL_ENV="$SAM3ENV" uv pip install -q -r 3_src/env_locks/sam3env_<chosen>.lock.txt
"$PY3" -c "import timm; assert timm.__version__=='1.0.28', timm.__version__"
```

⚠️ **哪一組是「對的」版本，取決於 run F 的 LB 結果**（08-30 跑，判準見
`0_README/HSOT_NEXT_ACTIONS.md` 的 D097 節）。在 F 出分之前**不要**改交付腳本的預設
——這是 D093 的教訓：改交付環境前先驗證修法，別猜著改。

## 已知限制

`sub_v078`（LB 0.71666）是 **chimera**：main／source 腿來自 v023/v012 時代的 run、
第三腿來自 08-22 的 run。因此**即使釘回 08-22 也不保證重現 0.71666**；
若 run F 落在 0.708 以下，下一步是去追 v023/v012 的環境與產生參數。
