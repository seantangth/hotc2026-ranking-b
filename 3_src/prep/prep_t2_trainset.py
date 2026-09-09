r"""T2 訓練資料準備 — 405 train zip → ViPT_HOT2023（HOT 標準）格式。

在 Lambda 實例跑（launch 腳本先 rclone 拉 zip 到本地，再呼叫本腳本）。

逐幀 GT 來源 = `1_data/raw/2026training.csv`（submission 格式，169,490 列，ID=`{seq}_{frame}`）——
不在 zip 內（假色/HSI zip 只含 init_rect.txt 首幀），必須從此 CSV 提取寫成每序列
`groundtruth_rect.txt`（tab 分隔 x,y,w,h，逐幀）。

輸出 HOT 目錄（ViPT 推論端 loader 直接吃；訓練端 loader 改現場 X2Cube 亦吃）：
    {OUT}/HSI-{VIS,NIR,RedNIR}/{name}/{NNNN}.png + groundtruth_rect.txt   # 原始 mosaic + GT
    {OUT}/HSI-{VIS,NIR,RedNIR}-FalseColor/{name}/{NNNN}.jpg + groundtruth_rect.txt
    {OUT}/hsi_{vis,nir,rednir}_{train,val}_split.txt                       # 序列 name 清單

**不預轉 .npy cube**（省 ~300GB）——改讓 ViPT `lib/train/dataset/hsi3d.py` 用現場 X2Cube
（複製推論端 `test_hsi_mgpus_all.py` 的 X2Cube / 或我方 io.py 邏輯）取代 `np.load(.npy)`。

update 修正優先：`update/HSI-*/{name}.zip` 存在則以它為準（75 序列的 HSI+假色替換包）。

用法：
    python prep_t2_trainset.py \
        --zips /workspace/hsot/data/raw_archive \  # 含 training/ 與 update/
        --gt   /workspace/hsot/data/2026training.csv \
        --val-split /workspace/hsot/data/val_split_v1.txt \
        --out  /workspace/hsot/data/t2_trainset
"""
from __future__ import annotations

import argparse
import csv
import zipfile
from collections import defaultdict
from pathlib import Path

MOD_DIR = {"vis": "HSI-VIS", "nir": "HSI-NIR", "rednir": "HSI-RedNIR"}


def load_gt(csv_path: Path) -> dict[str, list[tuple]]:
    """2026training.csv → {seq: [(frame, x, y, w, h), ...]}（按 frame 排序）。"""
    gt: dict[str, list[tuple]] = defaultdict(list)
    with open(csv_path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            seq, frame = row[0].rsplit("_", 1)
            gt[seq].append((int(frame), float(row[1]), float(row[2]), float(row[3]), float(row[4])))
    for s in gt:
        gt[s].sort()
    return gt


def pick_zip(zip_root: Path, sub: str, name: str) -> Path | None:
    """update 版優先（以 update 為準），否則原 training 版。"""
    up = zip_root / "update" / sub / f"{name}.zip"
    if up.exists():
        return up
    orig = zip_root / "training" / sub / f"{name}.zip"
    return orig if orig.exists() else None


IMG_EXT = (".png", ".jpg", ".jpeg")


def flatten_images(dst: Path) -> None:
    """把巢狀一層的影像搬到 `dst` 底下（`{name}/img/0001.png` → `{name}/0001.png`）。

    主辦方的 **update 修正包結構與原始包不同**：原始 zip 頂層是 `{name}/`，但 update 的
    頂層是 `img/`（實測 HSI-RedNIR 的 18 個 update zip 全部如此）→ 解出來變
    `{sub}/{name}/img/*.png`，而 ViPT loader 只 glob `{seq_path}/*.jpg`，會撲空並在
    訓練「隨機抽到該序列時」才崩潰（最難查的那種）。不論本次是否重新解壓都要執行。
    """
    if any(p.suffix.lower() in IMG_EXT for p in dst.iterdir() if p.is_file()):
        return                                   # 頂層已有影像＝結構正常
    # ⚠️ 巢狀可能不只一層：實測 vis-ant 的 update zip 是 `img/ant/*.jpg`（假色）與
    #    `img/img/*.png`（HSI）**兩層**。故用 rglob 一次撈到底，不要只處理單層。
    for f in sorted(dst.rglob("*")):
        if f.is_file() and f.suffix.lower() in IMG_EXT:
            target = dst / f.name
            if target.exists():
                continue                         # 同名衝突：保留先到者，不覆蓋
            f.rename(target)
    # 由深到淺清掉空目錄
    for d in sorted((p for p in dst.rglob("*") if p.is_dir()),
                    key=lambda p: len(p.parts), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()


def write_gt_txt(rows: list[tuple], out_txt: Path) -> None:
    """逐幀 x,y,w,h（tab 分隔）→ groundtruth_rect.txt（HOT 格式）。"""
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        for _frame, x, y, w, h in rows:
            f.write(f"{x:.0f}\t{y:.0f}\t{w:.0f}\t{h:.0f}\n")


def prep_one(seq: str, gt: dict, zip_root: Path, out_root: Path) -> str | None:
    """解一個序列的 HSI+假色 zip、寫 GT。回傳 None=成功，或錯誤訊息。"""
    mod = seq.split("-", 1)[0]
    name = seq.split("-", 1)[1]
    if seq not in gt:
        return f"{seq}: 無 GT"
    src_used = []
    for sub in (MOD_DIR[mod], MOD_DIR[mod] + "-FalseColor"):
        z = pick_zip(zip_root, sub, name)
        if z is None:
            return f"{seq}: 缺 {sub}/{name}.zip"
        src_used.append("upd" if "update" in str(z) else "orig")
        dst = out_root / sub / name
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            # 冪等：已解好就跳過（解 207 支 VIS 約 40 分鐘，重跑不該重付這個成本）。
            # 判準＝**只比對影像檔數**，兩邊口徑必須一致：zip 內本來就含一份
            # groundtruth_rect.txt（實測 car.zip 103 檔 = 102 png + gt），而我們又會自己寫一份
            # → 拿「全部檔案數」比會永遠差一個，導致每次都整批重解。
            want = sum(1 for m in zf.namelist() if m.lower().endswith(IMG_EXT))
            have = sum(1 for p in dst.rglob("*") if p.is_file() and p.name.lower().endswith(IMG_EXT))
            if not (want and have == want):
                # zip 內含頂層 `{name}/` 目錄（實測 car.zip → `car/0001.png`）→ 解到父層，
                # 否則會變成 `{sub}/{name}/{name}/*.png` 而 loader 的 glob 撲空。
                tops = {m.split("/", 1)[0] for m in zf.namelist() if m.strip("/")}
                zf.extractall(dst.parent if tops == {name} else dst)
        # ⚠️ flatten 必須在「跳過解壓」的情況下也執行——巢狀結構的目錄影像數是對的
        #    （rglob 會數到巢狀層），冪等判斷會判定「已完成」而跳過，若把 flatten 綁在
        #    解壓分支裡就永遠修不到既有的壞目錄。
        flatten_images(dst)
        # GT 寫兩處（HSI 與假色目錄各一份，確保 ViPT loader 從任一路徑都讀得到）
        write_gt_txt(gt[seq], dst / "groundtruth_rect.txt")
    # 終局檢查：不管 zip 結構怎麼變，最後只認「目錄底下實際有沒有影像」。
    # loader 是 glob 假色目錄的 *.jpg 並由此推導 HSI png 路徑 → 兩邊都必須非空，
    # 否則該序列會在訓練隨機抽中時才炸（fail_safe 還會把它包裝成「重試」）。
    n_png = len(list((out_root / MOD_DIR[mod] / name).glob("*.png")))
    n_jpg = len(list((out_root / (MOD_DIR[mod] + "-FalseColor") / name).glob("*.jpg")))
    if n_png == 0 or n_jpg == 0:
        return f"{seq}: 缺影像（png={n_png} jpg={n_jpg}）——排除"
    if n_png and n_png < len(gt[seq]):
        return f"{seq}: 幀數短缺 {n_png}<{len(gt[seq])}（{'/'.join(src_used)}；訓練以實際幀數為準，非錯誤）"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", required=True, help="含 training/ 與 update/ 的根目錄")
    ap.add_argument("--gt", required=True, help="2026training.csv")
    ap.add_argument("--val-split", required=True, help="val_split_v1.txt")
    ap.add_argument("--out", required=True, help="輸出根目錄（HOT 格式）")
    ap.add_argument("--limit", type=int, default=0, help="只處理前 N 序列（DRY 用）")
    ap.add_argument("--only", default="", help="逗號分隔序列名，只處理這些（DRY 用；如 vis-car,vis-coin）")
    args = ap.parse_args()

    gt = load_gt(Path(args.gt))
    val = set(Path(args.val_split).read_text().split())
    all_seqs = sorted(gt)
    if args.only:
        want = [s.strip() for s in args.only.split(",") if s.strip()]
        missing = [s for s in want if s not in gt]
        if missing:
            raise SystemExit(f"--only 指定的序列不在 GT：{missing}")
        all_seqs = want
    elif args.limit:
        all_seqs = all_seqs[: args.limit]
    zip_root, out = Path(args.zips), Path(args.out)
    print(f"{len(all_seqs)} 序列（GT 全量 {len(gt)}）；val_split {len(val)}；訓練 {len(all_seqs) - len(val & set(all_seqs))}")

    warns, failed = [], set()
    for i, seq in enumerate(all_seqs):
        w = prep_one(seq, gt, zip_root, out)
        if w:
            warns.append(w)
            # 「缺 zip」＝該序列根本沒有資料（如 gDrive 上的失效捷徑 rednir-backpack3）
            # → 必須排除在 split 外，否則 loader 會 glob 到空目錄而在訓練中途炸掉。
            # 「幀數短缺」不算失敗（D025：以實際幀數為準）。
            if ": 缺 " in w or "缺影像" in w or ": 無 GT" in w:
                failed.add(seq)
        if (i + 1) % 50 == 0:
            print(f"  處理 {i + 1}/{len(all_seqs)}")
    if failed:
        print(f"⚠️ {len(failed)} 支序列無資料，已排除於 split：{sorted(failed)}")
        all_seqs = [s for s in all_seqs if s not in failed]

    # split txt 格式須為 `HSI-{MOD}-FalseColor/{name}`（ViPT loader seq_path=root/此，已核官方 split）
    MOD_FC = {"vis": "HSI-VIS-FalseColor", "nir": "HSI-NIR-FalseColor", "rednir": "HSI-RedNIR-FalseColor"}
    for mod in ("vis", "nir", "rednir"):
        seqs_m = [s for s in all_seqs if s.startswith(mod + "-")]
        fc = MOD_FC[mod]
        tr = [f"{fc}/{s.split('-', 1)[1]}" for s in seqs_m if s not in val]
        va = [f"{fc}/{s.split('-', 1)[1]}" for s in seqs_m if s in val]
        (out / f"hsi_{mod}_train_split.txt").write_text("\n".join(tr) + "\n")
        (out / f"hsi_{mod}_val_split.txt").write_text("\n".join(va) + "\n")
        print(f"{mod}: train {len(tr)} / val {len(va)}")

    if warns:
        print(f"\n⚠️ {len(warns)} 序列告警（多為已知幀數短缺，非致命）：")
        for w in warns[:12]:
            print("  ", w)
    print(f"\n✓ 輸出 → {out}")
    print("下一步：ViPT loader 改現場 X2Cube（見 io.py），或本腳本加 --npy 預轉；然後 lambda_train_vipt.sh 訓練")


if __name__ == "__main__":
    main()
