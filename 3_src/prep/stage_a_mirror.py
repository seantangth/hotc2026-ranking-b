#!/usr/bin/env python3
"""HSOT Stage A：鏡像急用資料（假色全套 + 驗證集全套 + update 假色修正）到 gDrive。

在 Colab VM 上以 nohup 背景執行：
  nohup python3 /content/stage_a_mirror.py > /content/stage_a_stdout.log 2>&1 &

輸入（需先上傳到 /content/）：
  drive_manifest.json   — gdown --folder --json 抓的主辦方 Drive 清單
  2026training.csv      — 訓練 GT（驗證 zip 內容幀數用）
  sample_submisson.csv  — 測試幀清單（驗證 val 幀數用）
  rclone.conf           — 使用者 gdrive remote（cp 到 ~/.config/rclone/）

輸出：
  /content/mirror/...                 — 下載的資料（結構同 Drive）
  /content/stage_a_progress.log      — 進度（monitor 讀這個）
  /content/mirror/manifest_check.json — 驗證報告
  gdrive:HSOT/mirror/...              — rclone 上傳目的地
"""
import csv
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

MANIFEST = Path("/content/drive_manifest.json")
TRAIN_CSV = Path("/content/2026training.csv")
TEST_CSV = Path("/content/sample_submisson.csv")
OUT = Path("/content/mirror")
LOG = Path("/content/stage_a_progress.log")
DONE_LIST = Path("/content/stage_a_done.jsonl")
THREADS = 24
RCLONE_DEST = "gdrive:WHISPERS_2026_HyperSOT/1_data/mirror"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def load_frame_counts(path, col=0):
    counts = Counter()
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            seq = row[col].rsplit("_", 1)[0]
            counts[seq] += 1
    return counts


def direct_url(share_url):
    fid = parse_qs(urlparse(share_url).query)["id"][0]
    return f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"


def select_stage_a(items):
    """急用子集：全 validation + 訓練假色 + update 假色 + readme + 探測用最小 HSI zip。"""
    selected = []
    for it in items:
        p = it["path"]
        if p.startswith("validation/"):
            selected.append(it)
        elif p.startswith("training/") and "FalseColor" in p and "/update/" not in p:
            selected.append(it)
        elif p.startswith("training/update/") and "FalseColor" in p:
            selected.append(it)
        elif p.endswith("readme.txt"):
            selected.append(it)
    return selected


def select_probe(items, train_counts):
    """每模態挑幀數最少的 1 個訓練 HSI zip，供位深/X2Cube 驗證（D010 證據）。"""
    prefix = {"HSI-NIR": "nir", "HSI-RedNIR": "rednir", "HSI-VIS": "vis"}
    best = {}
    for it in items:
        parts = Path(it["path"]).parts
        if len(parts) == 3 and parts[0] == "training" and parts[1] in prefix and it["path"].endswith(".zip"):
            seq = f"{prefix[parts[1]]}-{Path(it['path']).stem}"
            n = train_counts.get(seq)
            if n and (parts[1] not in best or n < best[parts[1]][1]):
                best[parts[1]] = (it, n)
    return [v[0] for v in best.values()]


def download_one(session, item):
    rel = item["path"]
    dest = OUT / rel
    if dest.exists() and dest.stat().st_size > 0:
        return rel, dest.stat().st_size, "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err = None
    for attempt in range(4):
        try:
            with session.get(direct_url(item["url"]), stream=True, timeout=(15, 300)) as r:
                r.raise_for_status()
                if "text/html" in r.headers.get("content-type", ""):
                    raise RuntimeError("Drive returned an error page")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
            size = tmp.stat().st_size
            if size == 0:
                raise RuntimeError("empty file")
            tmp.rename(dest)
            return rel, size, "ok"
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    return rel, 0, f"FAILED: {last_err}"


def run_downloads(items, tag):
    total = len(items)
    done = failed = 0
    bytes_sum = 0
    t0 = time.time()
    with requests.Session() as session, ThreadPoolExecutor(THREADS) as pool, DONE_LIST.open("a") as donef:
        futures = [pool.submit(download_one, session, it) for it in items]
        for fut in as_completed(futures):
            rel, size, status = fut.result()
            if status.startswith("FAILED"):
                failed += 1
                log(f"FAIL {rel}: {status}")
            else:
                done += 1
                bytes_sum += size
                donef.write(json.dumps({"path": rel, "size": size}) + "\n")
            if (done + failed) % 500 == 0 or (done + failed) == total:
                rate = bytes_sum / max(time.time() - t0, 1) / 1e6
                log(f"{tag}: {done + failed}/{total} 完成 ({failed} 失敗), {bytes_sum / 1e9:.2f} GB, {rate:.0f} MB/s")
    return done, failed, bytes_sum


def probe_bit_depth():
    """檢查 val PNG 與 train HSI zip 位深 → D010 證據。"""
    import numpy as np
    from PIL import Image
    import zipfile

    report = {}
    for sub in ["validation/HSI-NIR", "validation/HSI-RedNIR"]:
        d = OUT / sub
        pngs = sorted(d.rglob("*.png"))[:2]
        for p in pngs:
            arr = np.array(Image.open(p))
            report[str(p.relative_to(OUT))] = {"dtype": str(arr.dtype), "shape": list(arr.shape), "max": int(arr.max())}
    for z in sorted((OUT / "_probe").glob("*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".png")][:1]
                for n in names:
                    with zf.open(n) as fh:
                        arr = np.array(Image.open(fh))
                    report[f"{z.name}::{n}"] = {"dtype": str(arr.dtype), "shape": list(arr.shape), "max": int(arr.max())}
        except Exception as e:  # noqa: BLE001
            report[str(z)] = {"error": str(e)}
    return report


def verify(test_counts):
    """驗證 validation 完整性：逐序列幀數 vs sample_submisson。"""
    issues = []
    stats = defaultdict(dict)
    folder_map = {"nir": ("HSI-NIR-FalseColor", "HSI-NIR"), "rednir": ("HSI-RedNIR-FalseColor", "HSI-RedNIR"), "vis": ("HSI-VIS-FalseColor", None)}
    for seq, n_expect in sorted(test_counts.items()):
        sensor, target = seq.split("-", 1)
        fc_folder, hsi_folder = folder_map[sensor]
        fc_dir = OUT / "validation" / fc_folder / target
        n_fc = len(list(fc_dir.glob("*.jpg"))) if fc_dir.exists() else 0
        stats[seq]["falsecolor"] = n_fc
        if n_fc != n_expect:
            issues.append(f"{seq}: falsecolor {n_fc} != 預期 {n_expect}")
        if not (fc_dir / "init_rect.txt").exists():
            issues.append(f"{seq}: 缺 init_rect.txt")
        if hsi_folder:
            hsi_dir = OUT / "validation" / hsi_folder / target
            n_hsi = len(list(hsi_dir.glob("*.png"))) if hsi_dir.exists() else 0
            stats[seq]["hsi"] = n_hsi
            if n_hsi != n_expect:
                issues.append(f"{seq}: HSI png {n_hsi} != 預期 {n_expect}")
        else:
            zpath = OUT / "validation" / "HSI-VIS" / f"{target}.zip"
            stats[seq]["hsi_zip"] = zpath.exists()
            if not zpath.exists():
                issues.append(f"{seq}: 缺 HSI-VIS zip")
    return issues, stats


def main():
    LOG.write_text("")
    log("Stage A 開始")
    items = json.load(MANIFEST.open())
    # manifest 路徑第一層若有共同 root，去掉
    roots = {Path(it["path"]).parts[0] for it in items}
    if len(roots) == 1 and next(iter(roots)) not in ("training", "validation"):
        for it in items:
            it["path"] = str(Path(*Path(it["path"]).parts[1:]))

    train_counts = load_frame_counts(TRAIN_CSV)
    test_counts = load_frame_counts(TEST_CSV)

    stage_a = select_stage_a(items)
    probes = select_probe(items, train_counts)
    probe_items = [{"url": it["url"], "path": f"_probe/{Path(it['path']).name}"} for it in probes]
    log(f"Stage A 檔案數: {len(stage_a)}, 探測 zip: {len(probe_items)}")

    d1, f1, b1 = run_downloads(stage_a, "mirror")
    d2, f2, b2 = run_downloads(probe_items, "probe")

    log("下載完成，開始驗證")
    issues, stats = verify(test_counts)
    bit_report = probe_bit_depth()
    check = {
        "downloaded": d1 + d2,
        "failed": f1 + f2,
        "bytes": b1 + b2,
        "validation_issues": issues,
        "bit_depth_probe": bit_report,
        "per_seq": {k: v for k, v in sorted(stats.items())},
    }
    (OUT / "manifest_check.json").write_text(json.dumps(check, indent=1, ensure_ascii=False))
    log(f"驗證: {len(issues)} 個問題; 位深探測 {len(bit_report)} 項")
    for issue in issues[:20]:
        log(f"  ISSUE: {issue}")
    for k, v in list(bit_report.items())[:8]:
        log(f"  BITDEPTH {k}: {v}")

    log("rclone 上傳 gDrive 開始")
    rc = subprocess.run(
        ["rclone", "copy", str(OUT), RCLONE_DEST, "--transfers", "16", "--checkers", "16", "--drive-chunk-size", "64M", "--stats-one-line", "--stats", "30s", "--log-level", "NOTICE"],
        capture_output=True, text=True, timeout=3600 * 3,
    )
    log(f"rclone exit={rc.returncode} tail={rc.stdout[-300:]} {rc.stderr[-300:]}")
    ok = f1 + f2 == 0 and len(issues) == 0 and rc.returncode == 0
    log(f"STAGE_A_DONE ok={ok} downloaded={d1 + d2} failed={f1 + f2} issues={len(issues)}")


if __name__ == "__main__":
    main()
