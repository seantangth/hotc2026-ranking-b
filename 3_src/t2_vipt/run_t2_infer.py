r"""T2 推論：合併後的 ckpt → 逐序列追蹤 → Kaggle 提交 CSV（一鍵）。

設計原則：**不重寫追蹤迴圈**，直接沿用上游 `tracking/test_hsi_mgpus_all.py` 的
`run_sequence`（含 X2Cube、RedNIR 丟末 band、依序列名自動判模態並傳 train_data_type、
首幀用 init_rect 初始化）——重寫等於製造與上游分歧的風險。我們只注入兩件上游寫死的東西：

  1. **checkpoint 路徑**：上游 `lib/test/parameter/vipt_hsi.py` 把它釘在
     `final_model/ViPT_all.pth.tar`（載入訓練產物那行被註解掉）→ 執行期 monkeypatch。
  2. **資料根目錄**：上游 `seq_home = '/your_path/validation'` 寫死 → 改用參數傳入
     （`run_sequence` 本來就吃 seq_home 參數，只有 __main__ 裡寫死）。

模態自動偵測靠資料夾名（HSI-VIS / HSI-NIR / HSI-RedNIR），符合 Ranking B「新資料夾丟進來
即跑、零人工步驟」的要求（COMPETITION_STRATEGY §6）。

用法：
    python run_t2_infer.py --repo <ViPT> --data <root>/validation \
        --ckpt merged.pth.tar --out sub.csv --sample sample_submisson.csv
    # 只跑本地驗證子集（提交前先過 val_split_v1 的紀律）：
    #   加 --only-seqs val_split_v1.txt
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

MOD_FOLDERS = ["HSI-VIS-FalseColor", "HSI-NIR-FalseColor", "HSI-RedNIR-FalseColor"]
MOD_PREFIX = {"HSI-VIS-FalseColor": "vis", "HSI-NIR-FalseColor": "nir",
              "HSI-RedNIR-FalseColor": "rednir"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="ViPT_HOT2023 根目錄")
    ap.add_argument("--data", required=True, help="含 HSI-*-FalseColor/{seq}/ 的資料根")
    ap.add_argument("--ckpt", required=True, help="合併後的 ckpt（merge_prompts.py 產出）")
    ap.add_argument("--out", required=True, help="輸出 submission CSV")
    ap.add_argument("--sample", default="", help="sample_submisson.csv（正式提交用；本地驗證可省略）")
    ap.add_argument("--dataset-mode", choices=["HOT23TEST", "HOT23VAL"], default="HOT23TEST",
                    help="TEST=首幀讀 init_rect.txt（官方測試集）；VAL=讀 groundtruth_rect.txt "
                         "首列（本地 val_split_v1，它是從 training 切出來的，只有逐幀 GT）")
    ap.add_argument("--label", default="run", help="本次執行標籤（結果落在 <repo>/results/<mode>_<label>/deep_all/）")
    ap.add_argument("--cfg", default="deep_all", help="experiments/vipt/<cfg>.yaml")
    ap.add_argument("--only-seqs", default="", help="只跑清單內序列（如 val_split_v1.txt，`vis-car` 格式）")
    ap.add_argument("--threads", type=int, default=1, help=">1 用多行程（單卡建議 1–2）")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)                       # 上游 save_path 是相對 cwd 的 ./results/...

    import lib.test.parameter.vipt_hsi as rgbt_params
    _orig = rgbt_params.parameters

    ckpt = str(Path(args.ckpt).resolve())

    def _patched(yaml_name: str, epoch):        # noqa: ANN001 — 對齊上游簽名
        p = _orig(yaml_name, epoch)
        p.checkpoint = ckpt                     # 上游釘死 ViPT_all.pth.tar，改指我們訓好的
        return p

    rgbt_params.parameters = _patched
    import tracking.test_hsi_mgpus_all as T
    T.rgbt_params.parameters = _patched         # 模組已 import 過的參照也要換掉

    # 序列清單：資料夾即真相（Ranking B 新資料直接可跑）
    data = Path(args.data).resolve()
    seq_list = []
    for folder in MOD_FOLDERS:
        d = data / folder
        if not d.is_dir():
            print(f"（無 {folder}，跳過）")
            continue
        seq_list += [f"{folder}/{p.name}" for p in sorted(d.iterdir()) if p.is_dir()]
    seq_list.sort()

    if args.only_seqs:
        want = set(Path(args.only_seqs).read_text().split())
        seq_list = [s for s in seq_list
                    if f"{MOD_PREFIX[s.split('/')[0]]}-{s.split('/')[1]}" in want]
        print(f"--only-seqs → 篩出 {len(seq_list)} 支")
    if not seq_list:
        raise SystemExit(f"🚨 {data} 底下找不到任何序列")

    # ⚠️ 上游把「yaml 檔名」與「結果資料夾名」綁成同一個參數 `yaml_name`
    #    （`params = parameters(yaml_name)` 會去開 experiments/vipt/<yaml_name>.yaml，
    #     而 save_path = ./results/<dataset_name>/<yaml_name>/）→ yaml_name 只能是真實的
    #    config 名。要區分不同執行，就把標籤放進 dataset_name：genConfig 是用
    #    `'HOT23VAL' in set_type` 這種 substring 判斷模式，所以加後綴不影響行為。
    dataset_name = f"{args.dataset_mode}_{args.label}"
    yaml_dir = args.cfg
    results_dir = repo / "results" / dataset_name / yaml_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"序列 {len(seq_list)} 支｜ckpt {Path(ckpt).name}｜輸出 {results_dir}")
    for i, seq in enumerate(seq_list, 1):
        print(f"[{i}/{len(seq_list)}] {seq}", flush=True)
        T.run_sequence(seq, str(data), dataset_name, yaml_dir, num_gpu=1, debug=0, epoch=0)

    # 轉檔（正式提交時附 --sample → 逐 ID 比對，格式錯誤在本地擋下，不浪費每日提交額度）
    conv = Path(__file__).with_name("results_to_submission.py")
    cmd = [sys.executable, str(conv), "--results", str(results_dir),
           "--out", str(Path(args.out).resolve())]
    if args.sample:
        cmd += ["--sample", str(Path(args.sample).resolve())]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
