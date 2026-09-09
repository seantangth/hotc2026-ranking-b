#!/usr/bin/env python3
"""HSOT Stage B：訓練 HSI（~310 GB）→ 訓練就緒 shards（每序列一個 .npz）→ gDrive。

v2（2026-08-03）：來源改為自家 Drive 的 raw_archive（伺服器端複製已先行），
用認證 rclone 下載（走自己 API 配額，匿名端點限流問題根治）；
uint16 全保真（D010 修訂：val 有 max=25 的極暗場景，divisor 量化會毀掉暗部訊號）。

流程（串流，VM 磁碟峰值受控）：
  rclone 從 raw_archive 抓 zip（update 優先；未複製到就等）→ zip 內直接解碼 mosaic PNG →
  修正版 X2Cube → uint16 直存 → 流式寫入 .npz（每幀一個 member，np.load 可懶讀）→
  rclone 上傳 shard → 刪本地 zip。

在 Colab VM 上背景執行：
  nohup python3 /content/stage_b_shards.py > /content/stage_b_stdout.log 2>&1 &

可斷點續傳：啟動時 rclone lsf 既有 shards 自動跳過。
決策狀態存 /content/stage_b_state.json，續跑沿用（一致性保證）。
"""
import csv
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import zipfile
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
import numpy.lib.format as npf

BASE = Path(os.environ.get("HSOT_BASE", "/content"))  # Colab=/content, RunPod=/workspace
MANIFEST = BASE / "drive_manifest.json"
TRAIN_CSV = BASE / "2026training.csv"
LOG = BASE / "stage_b_progress.log"
STATE = BASE / "stage_b_state.json"
DL_DIR = BASE / "stage_b_dl"
SHARD_DIR = BASE / "stage_b_shards"
GDRIVE_SHARDS = "gdrive:WHISPERS_2026_HyperSOT/1_data/shards/train"
GDRIVE_RAW = "gdrive:WHISPERS_2026_HyperSOT/1_data/raw_archive"
DISK_MIN_FREE_GB = 8  # RunPod CPU pod 只有 20GB container disk（volumeInGb 對 CPU pod 無效）
COMPRESS_LEVEL = 1  # 單執行緒 deflate 是吞吐瓶頸；level 1 檔案僅大 ~10% 但快 2-3 倍
RAW_WAIT_MAX = 30  # 等 raw_archive 伺服器端複製到位的最大輪數（×120s）

MODALITY = {  # sensor -> (Drive 資料夾, mosaic block, raw bands)
    "nir": ("HSI-NIR", 5, 25),
    "rednir": ("HSI-RedNIR", 4, 16),
    "vis": ("HSI-VIS", 4, 16),
}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def free_gb(path=None):
    return shutil.disk_usage(str(path or BASE)).free / 1e9


def x2cube(img, block, bands):
    """修正版 X2Cube（官方版 reshape 寫死 //4，NIR 5×5 會錯）。band 順序 = mosaic 區塊 row-major，與官方一致。"""
    M, N = img.shape
    h, w = M // block, N // block
    img = img[: h * block, : w * block]
    cube = img.reshape(h, block, w, block).transpose(0, 2, 1, 3).reshape(h, w, block * block)
    return cube[:, :, :bands]


# ---- multiprocessing worker（每個 worker 自己開 zip handle）----
_worker_zip = None
_worker_cfg = None


def _init_worker(zip_path, block, bands):
    global _worker_zip, _worker_cfg
    cv2.setNumThreads(0)  # 避免與 multiprocessing 超訂
    _worker_zip = zipfile.ZipFile(zip_path)
    _worker_cfg = (block, bands)


def frame_key(name):
    """數字感知排序：0001.png 與 1.png 兩種命名都正確。"""
    stem = Path(name).stem
    return (0, int(stem), "") if stem.isdigit() else (1, 0, stem)


def _decode_one(name):
    block, bands = _worker_cfg
    raw = _worker_zip.read(name)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return name, None
    if img.ndim == 3:  # 保險：不應發生（mosaic 是單通道）
        img = img[:, :, 0]
    return name, x2cube(img, block, bands)


def fetch_zip(remote_rel, dest):
    """從自家 raw_archive 認證下載；伺服器端複製還沒到位就等。"""
    src = f"{GDRIVE_RAW}/{remote_rel}"
    for attempt in range(RAW_WAIT_MAX):
        r = rclone(["copyto", src, str(dest), "--drive-chunk-size", "128M"], timeout=3600)
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return True, None
        not_found = "not found" in (r.stderr or "").lower() or "directory not found" in (r.stderr or "").lower()
        if not_found:
            time.sleep(120)  # 等 B1 複製工作補上
            continue
        time.sleep(20 * min(attempt + 1, 5))
    return False, (r.stderr or "")[-200:]


def rclone(args, timeout=1800):
    return subprocess.run(["rclone", *args], capture_output=True, text=True, timeout=timeout)


def load_tasks():
    """405 訓練序列 → 下載路徑（update 版優先）+ 預期幀數。"""
    counts = Counter()
    with TRAIN_CSV.open() as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            counts[row[0].rsplit("_", 1)[0]] += 1

    items = json.load(MANIFEST.open())
    url_by_path = {it["path"]: it["url"] for it in items}
    tasks = []
    missing = []
    for seq, n in sorted(counts.items()):
        sensor, target = seq.split("-", 1)
        folder = MODALITY[sensor][0]
        canonical = f"training/{folder}/{target}.zip"
        update = f"training/update/{folder}/{target}.zip"
        path = update if update in url_by_path else canonical
        if path not in url_by_path:
            missing.append(seq)
            continue
        tasks.append({"seq": seq, "sensor": sensor, "path": path, "url": url_by_path[path], "n_expect": n, "from_update": path == update})
    if missing:
        log(f"ISSUE 找不到 zip 的序列: {missing}")
    return tasks


def existing_shards():
    done = set()
    for sensor in MODALITY:
        r = rclone(["lsf", f"{GDRIVE_SHARDS}/{sensor}/"], timeout=300)
        if r.returncode == 0:
            done |= {f"{sensor}-{Path(line).stem}" for line in r.stdout.split() if line.endswith(".npz")}
    return done


def decide_quant(zip_path, block, bands, sensor, state):
    """首見模態：抽 ≤16 幀決定 divisor 與 RedNIR 斷帶。之後固定沿用。"""
    if sensor in state["divisor"]:
        return
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted((n for n in zf.namelist() if n.lower().endswith(".png")), key=frame_key)
        sample = names[:: max(len(names) // 16, 1)][:16]
        maxv = 0
        band_last_max = 0
        dtype = None
        for n in sample:
            img = cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_UNCHANGED)
            dtype = str(img.dtype)
            cube = x2cube(img, block, bands)
            maxv = max(maxv, int(cube.max()))
            band_last_max = max(band_last_max, int(cube[:, :, -1].max()))
    # D010 修訂：一律全保真。uint16 來源直存 uint16（val 有 max=25 的極暗場景，量化必毀暗部）
    div = 1 if dtype == "uint8" else 0
    state["divisor"][sensor] = div
    state["src_dtype"][sensor] = dtype
    state["src_max"][sensor] = maxv
    if sensor == "rednir":
        state["rednir_drop_last"] = band_last_max == 0
        log(f"RedNIR band16 max={band_last_max} → drop_last={state['rednir_drop_last']}")
    STATE.write_text(json.dumps(state, indent=1))
    log(f"量化決策 {sensor}: dtype={dtype} max={maxv} divisor={div}")


def pack_sequence(task, state):
    seq, sensor = task["seq"], task["sensor"]
    _folder, block, bands = MODALITY[sensor]
    zip_path = DL_DIR / f"{Path(task['path']).name}"
    decide_quant(zip_path, block, bands, sensor, state)
    div = state["divisor"][sensor]
    out_bands = bands
    if sensor == "rednir" and state.get("rednir_drop_last"):
        out_bands = bands - 1

    with zipfile.ZipFile(zip_path) as zf:
        names = sorted((n for n in zf.namelist() if n.lower().endswith(".png")), key=frame_key)
    n_frames = len(names)
    if n_frames != task["n_expect"]:
        log(f"ISSUE {seq}: zip 幀數 {n_frames} != CSV {task['n_expect']}（照常打包，訓練時以實際幀數為準）")

    shard = SHARD_DIR / sensor / f"{seq.split('-', 1)[1]}.npz"
    shard.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # 容器內 cpu_count() 可能回報宿主核心數（RunPod 上看到 256）→ 用 env 明確指定
    n_workers = int(os.environ.get("HSOT_WORKERS", "0")) or max(min(cpu_count(), 16) - 1, 1)
    written = 0
    clipped = 0
    with zipfile.ZipFile(shard, "w", zipfile.ZIP_DEFLATED, compresslevel=COMPRESS_LEVEL) as out_zf:
        meta = {
            "seq": seq, "sensor": sensor, "block": block, "raw_bands": bands, "bands": out_bands,
            "divisor": div, "src_dtype": state["src_dtype"][sensor], "n_frames": n_frames,
            "frame_names": [Path(n).stem for n in names],
        }
        out_zf.writestr("meta.json", json.dumps(meta))
        with Pool(n_workers, initializer=_init_worker, initargs=(str(zip_path), block, bands)) as pool:
            for name, cube in pool.imap(_decode_one, names, chunksize=4):
                if cube is None:
                    log(f"ISSUE {seq}: 解碼失敗 {name}")
                    continue
                if sensor == "rednir" and out_bands < bands:
                    cube = cube[:, :, :out_bands]
                if div == 1:
                    clipped += int((cube > 255).sum())
                    arr = cube.clip(0, 255).astype(np.uint8)
                elif div > 1:
                    clipped += int((cube >= 256 * div).sum())
                    arr = (cube.astype(np.uint32) // div).clip(0, 255).astype(np.uint8)
                else:
                    arr = cube.astype(np.uint16)
                with out_zf.open(f"f{Path(name).stem}.npy", "w", force_zip64=True) as fh:
                    npf.write_array(fh, arr, allow_pickle=False)
                written += 1
    dt = time.time() - t0
    mb = shard.stat().st_size / 1e6
    clip_note = f", 削頂像素 {clipped}" if clipped else ""
    log(f"{seq}: {written}/{n_frames} 幀 → {mb:.0f} MB, {dt:.0f}s ({written / max(dt, 1):.1f} f/s){clip_note}")
    return shard


def main():
    LOG.write_text("")
    DL_DIR.mkdir(exist_ok=True)
    SHARD_DIR.mkdir(exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {"divisor": {}, "src_dtype": {}, "src_max": {}}
    tasks = load_tasks()
    done = existing_shards()
    todo = [t for t in tasks if t["seq"] not in done]
    part = os.environ.get("STAGE_B_PART")  # 多 session 分工，如 "0/2"、"1/2"
    if part:
        idx, total = map(int, part.split("/"))
        todo = [t for i, t in enumerate(todo) if i % total == idx]
        log(f"分工模式 {part}: 本機負責 {len(todo)} 序列")
    log(f"Stage B 開始: 總 {len(tasks)} 序列, 已完成 {len(done)}, 待處理 {len(todo)}, workers={cpu_count()}")

    dl_q: queue.Queue = queue.Queue(maxsize=3)
    fail: list[str] = []

    def downloader():
        for t in todo:
            while free_gb() < DISK_MIN_FREE_GB:
                time.sleep(30)
            dest = DL_DIR / Path(t["path"]).name
            if not dest.exists():
                ok, err = fetch_zip(t["path"], dest)
                if not ok:
                    log(f"FAIL 下載 {t['seq']}: {err}")
                    fail.append(t["seq"])
                    continue
            dl_q.put(t)
        dl_q.put(None)

    threading.Thread(target=downloader, daemon=True).start()

    # 上傳背景化：與下一序列的轉換重疊（同步上傳曾使 IO 佔週期 75%）
    # 兩條上傳線：Drive 限速為每連線 ~10 MB/s，雙線並行貼近 20 MB/s（上傳是尾段瓶頸）
    N_UPLOADERS = 3
    up_q: queue.Queue = queue.Queue(maxsize=3)
    stats = {"ok": 0}

    def uploader():
        while True:
            item = up_q.get()
            if item is None:
                up_q.task_done()
                break
            shard, sensor, target, zip_local = item
            up_ok = False
            for attempt in range(6):
                try:
                    r = rclone(["moveto", str(shard), f"{GDRIVE_SHARDS}/{sensor}/{target}.npz", "--drive-chunk-size", "128M"], timeout=3600)
                except subprocess.TimeoutExpired:
                    log(f"上傳逾時（1h）retry {attempt + 1}/6: {sensor}-{target}")
                    continue
                except Exception as e:  # noqa: BLE001 — uploader 執行緒絕不能死（曾因未接 TimeoutExpired 全線死鎖）
                    log(f"上傳例外 retry {attempt + 1}/6: {sensor}-{target}: {e!r}")
                    time.sleep(30)
                    continue
                if r.returncode == 0:
                    up_ok = True
                    break
                time.sleep(30 * (attempt + 1))
            if up_ok:
                zip_local.unlink(missing_ok=True)  # raw 已由伺服器端複製歸檔，本地即棄
                stats["ok"] += 1
                if stats["ok"] % 10 == 0:
                    log(f"進度: {stats['ok']}/{len(todo)} 上傳完成, 磁碟餘 {free_gb():.0f} GB")
            else:
                log(f"FAIL 上傳 shard {sensor}-{target}: {r.stderr[-150:]}")
                fail.append(f"{sensor}-{target}")
                shard.unlink(missing_ok=True)
            up_q.task_done()

    for _ in range(N_UPLOADERS):
        threading.Thread(target=uploader, daemon=True).start()

    while True:
        t = dl_q.get()
        if t is None:
            break
        zip_local = DL_DIR / Path(t["path"]).name
        # 空間閘門：shard ≈ 1.2× zip 大小，不足就等 uploader 清空間（防 ENOSPC 丟序列）
        need_gb = zip_local.stat().st_size * 1.25 / 1e9 + 2 if zip_local.exists() else 4
        waited = 0
        while free_gb() < need_gb and waited < 1800:
            time.sleep(20)
            waited += 20
        if waited:
            log(f"{t['seq']}: 等待磁碟空間 {waited}s（需 {need_gb:.1f} GB）")
        try:
            shard = pack_sequence(t, state)
        except Exception as e:  # noqa: BLE001
            log(f"FAIL 打包 {t['seq']}: {e!r}")
            fail.append(t["seq"])
            zip_local.unlink(missing_ok=True)
            sensor, target = t["seq"].split("-", 1)
            (SHARD_DIR / sensor / f"{target}.npz").unlink(missing_ok=True)  # 清半成品防磁碟洩漏
            continue
        sensor, target = t["seq"].split("-", 1)
        up_q.put((shard, sensor, target, zip_local))

    up_q.join()
    for _ in range(N_UPLOADERS):
        up_q.put(None)
    log(f"STAGE_B_DONE ok={len(fail) == 0} packed={stats['ok']}/{len(todo)} failed={len(fail)} {fail[:20]}")


if __name__ == "__main__":
    main()
