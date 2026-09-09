#!/usr/bin/env python3
"""prep_rankingb_frames_v1 — 9/7 官方發佈 → `<FRAMES_ROOT>/<seq>/*.jpg + init_rect.txt`（無 GT）。

【為什麼要有這支】RUNBOOK 開機後第 3 步「資料 ingestion」是**全部 6 次演練中唯一從未跑過**的一步
（09-04 稽核 G15）。它目前只活在 08-06 的實驗腳本裡（`unzip -q` 一行），而 9/7 是三天硬窗口：
現場手寫轉檔的風險不在「寫不出來」，在**寫錯了不會當場報錯**——
`run_ranking_b.py` 的 preflight 雖然會擋 init_rect 超出影像與非數字檔名（G15 已補），
但那是在**已經產出目錄之後**才檢查；而座標慣例錯誤（x1y1x2y2 被當成 xywh）是
**唯一會燒掉整輪 4.5 小時**的失敗型態。這支把檢查前移到轉檔當下，並留下 manifest 供事後對帳。

【設計原則】**輸入格式未知、輸出契約已知** ⇒ 輸入端寬鬆自動偵測，輸出端嚴格 fail-closed。
官方 2026 的發佈形狀沒有人看過（歷年有：每序列一個 zip／一個大 zip 內含序列目錄／直接給目錄），
所以三種都吃；但輸出一定是 track_t1.py 認得的那一種，且每一項契約都當場驗。

【輸出契約（對照 `track_t1.py:52` read_init 與 `run_ranking_b.py:723` preflight）】
  <FRAMES_ROOT>/<seq>/0001.jpg …          檔名 stem 必須可 int()（track_t1 依 int(stem) 排序後
                                          的**位置**對應 sample 列序，非數字即 BLOCK）
  <FRAMES_ROOT>/<seq>/init_rect.txt       首幀框，寫成 `x y w h`（空白分隔；read_init 對
                                          逗號亦容錯，但統一寫空白＝與 Ranking A 同慣例）
  <OUT>/INGEST_MANIFEST.json              逐序列幀數／init 來源／是否重編號／影像尺寸

【三道當場驗的 fail-closed（依嚴重度）】
  1. **init 座標慣例**：x+w ≤ W 且 y+h ≤ H（放寬 1px 容忍）。抓的是 (x1,y1,x2,y2) 被誤當
     (x,y,w,h)——那時 x+w 幾乎必然爆寬。這條錯了，整輪 4.5 小時的推論全部作廢。
  2. **檔名 stem 可 int()**：否則 track_t1 的排序對應會靜默錯位。非數字時預設拒絕，
     `--renumber` 才允許改名（依自然序 1..N 重編，映射寫進 manifest）。
  3. **sample 契約**（給 `--sample` 時）：序列集合與逐序列幀數必須完全相符。

【09-07 補：官方佈局實看後新增三件】
  4. **模態前綴**：官方把模態放在上層資料夾名（`HSI-NIR-Falsecolor/<seq>/`），管線卻靠序列名
     前綴（`nir-`）判模態，沒前綴時 `quality_head_v1.modality()` 會靜默全當 VIS。
     ⇒ 自動從模態資料夾名推前綴；`--source` 可指 `ranking/` 根或單一模態資料夾；`--prefix` 可明示。
  5. **容器目錄 fail-closed**：影像在更深一層的目錄以前會被 rglob 整層吞進來、同名幀互相覆蓋、
     靜默合併成一支「序列」——現在直接 FATAL。16-bit PNG（mosaic 原始資料）也擋。
  6. **`--write-sample`**：主辦方私有集不一定附 sample_submission.csv，而 run_ranking_b.py
     必填 ⇒ 從驗證通過的序列產一份（`<seq>_<1..n>`、序列依名排序，與 Ranking A 官方檔同型）。

用法：
    # 官方佈局（ranking/ 根，或任一 HSI-<模態>-Falsecolor/）：
    python 3_src/prep/prep_rankingb_frames_v1.py \
        --source <ranking/> --frames-out <FRAMES_ROOT> --write-sample <FRAMES_ROOT>/sample_submission.csv
    # 其他形狀：
    python 3_src/prep/prep_rankingb_frames_v1.py \
        --source <官方解開/下載後的目錄> --frames-out <FRAMES_ROOT> \
        [--sample sample_submission.csv] [--init-dir <另附的 init 目錄>] [--renumber] [--prefix nir-]

不做的事：不下載、不解密、不碰 GT（groundtruth_rect.txt 只在缺 init_rect 時當首幀來源，不複製進 frames-root）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}
INIT_NAMES = {"init_rect.txt", "groundtruth_rect.txt", "init.txt", "groundtruth.txt"}
BOUND_TOL = 1.0  # 影像邊界容忍（px）；官方標註偶有貼邊四捨五入

# ── 官方模態佈局（09-07 實看 gDrive `ranking/`）─────────────────────────────
#   ranking/HSI-NIR/<seq>/0001.png…            16-bit mosaic（管線不吃）
#   ranking/HSI-NIR-Falsecolor/<seq>/0001.jpg…  假色 + init_rect.txt + groundtruth_rect.txt
#   （RedNIR 同型；VIS 的假色資料夾叫 HSI-VIS-FalseColor，大小寫不一）
# 管線的序列名慣例是 `<模態>-<seq>`（nir-/rednir-/vis-），而官方把模態放在**上層資料夾名**
# ⇒ 沒有前綴時 `quality_head_v1.modality()` 會把所有序列當 VIS（靜默、無 BLOCK），
#    所以這裡從資料夾名推前綴，推不出來就拒絕（--prefix 可明示）。
MODALITY_DIR_RE = re.compile(r"^HSI-(NIR|RedNIR|VIS)(-False[-_]?colou?r)?$", re.IGNORECASE)
MODALITY_PREFIX = {"nir": "nir-", "rednir": "rednir-", "vis": "vis-"}


def modality_dir_info(name: str) -> tuple[str, bool] | None:
    """`HSI-NIR-Falsecolor` → ("nir-", True)；`HSI-NIR` → ("nir-", False)；其他 → None。"""
    m = MODALITY_DIR_RE.match(name.strip())
    if not m:
        return None
    return MODALITY_PREFIX[m.group(1).lower()], bool(m.group(2))


def png_bit_depth(data: bytes) -> int | None:
    """PNG 的 IHDR bit depth（mosaic 是 16）；非 PNG 回 None。"""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 25:
        return data[24]
    return None


# ── 影像尺寸：優先用 PIL；沒有 PIL 就讀 JPEG/PNG 檔頭 ────────────────────────
def image_size(data: bytes) -> tuple[int, int] | None:
    """回傳 (width, height)；無法判定回 None（呼叫端據此降級為警告而非 BLOCK）。"""
    try:
        import io

        from PIL import Image  # type: ignore

        with Image.open(io.BytesIO(data)) as im:
            return int(im.width), int(im.height)
    except Exception:
        pass
    # PNG：IHDR 固定在第 16 byte 起
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    # JPEG：掃 SOFn 標記
    if data[:2] == b"\xff\xd8":
        i = 2
        n = len(data)
        while i + 9 < n:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seg_len = int.from_bytes(data[i + 2:i + 4], "big")
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h = int.from_bytes(data[i + 5:i + 7], "big")
                w = int.from_bytes(data[i + 7:i + 9], "big")
                return w, h
            i += 2 + seg_len
    return None


def parse_box(text: str) -> tuple[float, float, float, float] | None:
    """取第一列非空的四個數；空白或逗號分隔皆可（VOT 慣例是逗號，Ranking A 是空白）。"""
    for line in text.splitlines():
        tokens = [t for t in re.split(r"[,\s]+", line.strip()) if t]
        if len(tokens) >= 4:
            try:
                return tuple(float(t) for t in tokens[:4])  # type: ignore[return-value]
            except ValueError:
                continue
    return None


def natural_key(name: str):
    """自然序：抽出結尾數字當主鍵，抓不到就退回字串比較（重編號時保序用）。"""
    m = re.search(r"(\d+)(?=\D*$)", Path(name).stem)
    return (0, int(m.group(1)), name) if m else (1, 0, name)


# ── 來源抽象：一支序列 ＝ 一組 (幀名 → bytes) ＋ 可選的 init 文字 ─────────────
class SeqSource:
    def __init__(self, name: str, origin: str):
        self.name = name
        self.origin = origin

    def frames(self) -> list[str]:
        raise NotImplementedError

    def read(self, member: str) -> bytes:
        raise NotImplementedError

    def init_text(self) -> str | None:
        raise NotImplementedError


class ZipSeq(SeqSource):
    def __init__(self, name: str, zf: zipfile.ZipFile, members: list[zipfile.ZipInfo],
                 init_member: zipfile.ZipInfo | None, origin: str):
        super().__init__(name, origin)
        self._zf = zf
        self._members = {Path(m.filename).name: m for m in members}
        self._init = init_member

    def frames(self) -> list[str]:
        return sorted(self._members, key=natural_key)

    def read(self, member: str) -> bytes:
        return self._zf.read(self._members[member])

    def init_text(self) -> str | None:
        if self._init is None:
            return None
        return self._zf.read(self._init).decode("utf-8-sig", errors="replace")


class DirSeq(SeqSource):
    def __init__(self, name: str, root: Path, origin: str):
        super().__init__(name, origin)
        self._root = root
        self._files = {p.name: p for p in root.rglob("*")
                       if p.is_file() and p.suffix.lower() in IMG_EXTS}
        self._init: Path | None = None
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.name.lower() in INIT_NAMES:
                # init_rect 優先於 groundtruth
                if self._init is None or "init" in p.name.lower():
                    self._init = p

    def frames(self) -> list[str]:
        return sorted(self._files, key=natural_key)

    def read(self, member: str) -> bytes:
        return self._files[member].read_bytes()

    def init_text(self) -> str | None:
        return self._init.read_text(encoding="utf-8-sig", errors="replace") if self._init else None


def zip_sequences(path: Path) -> list[SeqSource]:
    """一個 zip：可能是一支序列，也可能內含多個序列目錄。兩種都吃。"""
    zf = zipfile.ZipFile(path)
    images = [m for m in zf.infolist()
              if not m.is_dir() and Path(m.filename).suffix.lower() in IMG_EXTS]
    if not images:
        raise ValueError(f"{path.name}: zip 內找不到影像")
    inits = [m for m in zf.infolist()
             if not m.is_dir() and Path(m.filename).name.lower() in INIT_NAMES]

    # 以「影像所在目錄」分群；只有一群 ⇒ 整個 zip 是一支序列
    groups: dict[str, list[zipfile.ZipInfo]] = {}
    for m in images:
        groups.setdefault(str(Path(m.filename).parent), []).append(m)

    def _named(parent: str, fallback: str) -> str | None:
        """zip 內 `…/HSI-NIR-Falsecolor/<seq>/` 的序列名 ＝ 前綴＋<seq>；mosaic 資料夾回 None（略過）。

        Drive「下載資料夾為 zip」就是這個形狀（09-07 實看 `ranking/…`），與目錄路徑同一個前綴洞。
        """
        p = Path(parent)
        seq = p.name or fallback
        info = modality_dir_info(p.parent.name) if p.name else None
        if info is None:
            return seq
        pfx, is_fc = info
        return pfx + seq if is_fc else None

    skipped: list[str] = []
    if len(groups) == 1:
        parent = next(iter(groups))
        init = None
        for m in sorted(inits, key=lambda x: ("init" not in Path(x.filename).name.lower(), x.filename)):
            init = m
            break
        name = _named(parent, path.stem)
        if name is None:
            raise ValueError(f"{path.name}: 內容在 mosaic 資料夾下（{parent}），管線只吃假色")
        return [ZipSeq(name, zf, images, init, f"zip:{path.name}")]

    out: list[SeqSource] = []
    for parent, members in sorted(groups.items()):
        name = _named(parent, path.stem)
        if name is None:
            skipped.append(parent)
            continue
        cand = [m for m in inits if str(Path(m.filename).parent) == parent]
        init = None
        for m in sorted(cand, key=lambda x: ("init" not in Path(x.filename).name.lower(), x.filename)):
            init = m
            break
        out.append(ZipSeq(name, zf, members, init, f"zip:{path.name}!{parent}"))
    if skipped:
        print(f"[discover] {path.name}: 略過 mosaic 目錄 {len(skipped)} 個（例 {skipped[0]}）")
    if not out:
        raise ValueError(f"{path.name}: 只有 mosaic 目錄、沒有 *-Falsecolor")
    # ⚠️ Drive 會把大下載切成 -001.zip／-002.zip；同一支序列若跨兩個分割檔，
    # 會在 discover 的 _check_dup 以「序列名重複」FATAL——那是 fail-closed，不是 bug：先合併分割檔再跑。
    return out


def _has_images(d: Path) -> bool:
    return any(p.suffix.lower() in IMG_EXTS for p in d.iterdir() if p.is_file())


def _image_subdirs(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_dir() and _has_images(p))


def _dir_sequences(container: Path, prefix: str, origin_tag: str) -> list[SeqSource]:
    """container 的每個含影像子目錄 ＝ 一支序列（名前加 prefix）。"""
    out: list[SeqSource] = []
    for d in sorted(p for p in container.iterdir() if p.is_dir()):
        if _has_images(d):
            out.append(DirSeq(prefix + d.name, d, f"{origin_tag}:{d.name}"))
            continue
        nested = _image_subdirs(d)
        if len(nested) == 1 and modality_dir_info(d.name) is None and not _image_subdirs(nested[0]):
            # `<seq>/img/0001.jpg` 這種只多一層的形狀不可能發生合併，接受；序列名用外層
            out.append(DirSeq(prefix + d.name, nested[0], f"{origin_tag}:{d.name}/{nested[0].name}"))
            continue
        if nested and modality_dir_info(d.name) is None:
            # 影像在更深一層：這是容器不是序列。以前會把整層 rglob 進來、同名幀互相覆蓋、
            # 靜默合併成一支「序列」——這正是 9/7 不可以發生的失敗型態。
            raise SystemExit(
                f"FATAL: {d} 本身沒有影像、但含 {len(nested)} 個有影像的子目錄（例 {nested[0].name}）"
                "——它是容器不是序列。請把 --source 指到序列目錄的上一層，"
                "或用官方模態佈局（HSI-<模態>-Falsecolor/<seq>/）。")
    return out


def discover(source: Path, prefix: str | None = None) -> list[SeqSource]:
    """自動偵測：每序列一 zip／單一大 zip／已解開的目錄／官方模態佈局。

    prefix=None ⇒ 依資料夾名推（官方佈局），推不出來就不加；
    prefix="" ⇒ 明示不加；prefix="nir-" ⇒ 明示加。
    """
    if source.is_file() and source.suffix.lower() == ".zip":
        seqs = zip_sequences(source)
        if prefix:
            for s in seqs:
                s.name = prefix + s.name
        return seqs
    if not source.is_dir():
        raise SystemExit(f"FATAL: --source 不是目錄也不是 zip：{source}")

    # (a) --source 本身就是一個模態資料夾（HSI-NIR-Falsecolor/）
    info = modality_dir_info(source.name)
    if info is not None:
        auto_prefix, is_fc = info
        if not is_fc:
            raise SystemExit(
                f"FATAL: {source.name} 是 16-bit mosaic 資料夾；管線只吃假色，"
                f"請指到 {source.name}-Falsecolor（或用 --prefix 明示且確認影像是假色）")
        return _dir_sequences(source, auto_prefix if prefix is None else prefix,
                              f"dir:{source.name}")

    # (b) --source 下面是多個模態資料夾（官方 ranking/ 根）
    mod_dirs = [(d, modality_dir_info(d.name)) for d in sorted(source.iterdir()) if d.is_dir()]
    mod_dirs = [(d, i) for d, i in mod_dirs if i is not None]
    if mod_dirs:
        out: list[SeqSource] = []
        skipped_mosaic: list[str] = []
        for d, (auto_prefix, is_fc) in mod_dirs:
            if not is_fc:
                skipped_mosaic.append(d.name)
                continue
            out.extend(_dir_sequences(d, auto_prefix if prefix is None else prefix,
                                      f"dir:{d.name}"))
        if skipped_mosaic:
            print(f"[discover] 略過 mosaic 資料夾（管線不吃 16-bit）：{skipped_mosaic}")
        if not out:
            raise SystemExit(f"FATAL: {source} 下只有 mosaic 資料夾、沒有 *-Falsecolor")
        _check_dup(out)
        return out

    # (c) 舊形狀：zip 們 ＋ 直接的序列目錄
    zips = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() == ".zip")
    out = []
    for z in zips:
        out.extend(zip_sequences(z))
    out.extend(_dir_sequences(source, "", "dir"))
    if prefix:
        for s in out:
            s.name = prefix + s.name

    if not out:
        raise SystemExit(f"FATAL: {source} 下找不到任何 zip 或含影像的序列目錄")
    _check_dup(out)
    return out


def _check_dup(seqs: list[SeqSource]) -> None:
    names = [s.name for s in seqs]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise SystemExit(f"FATAL: 序列名重複（同名 zip 與目錄並存？）：{sorted(dup)[:5]}")


def write_sample_csv(path: Path, manifest: dict[str, dict]) -> int:
    """從已驗證的序列產 sample_submission.csv：`<seq>_<i>`（i 從 1 起、依幀序）、序列依名排序。

    這與 Ranking A 官方檔同型（ID,x,y,width,height；幀號＝1-based 位置），
    也是 run_ranking_b.py `_sample_contract` 與 track_t1 `rows_from_boxes` 的隱性契約。
    """
    import csv

    n = 0
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ID", "x", "y", "width", "height"])
        for seq in sorted(manifest):
            for i in range(1, manifest[seq]["n_frames"] + 1):
                w.writerow([f"{seq}_{i}", 0, 0, 0, 0])
                n += 1
    return n


def load_sample_counts(path: Path) -> dict[str, int]:
    import csv

    counts: dict[str, int] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            seq = row["ID"].rsplit("_", 1)[0]
            counts[seq] = counts.get(seq, 0) + 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path,
                    help="官方發佈的目錄（內含每序列 zip 或已解開的序列目錄），或單一大 zip")
    ap.add_argument("--frames-out", required=True, type=Path, help="輸出 <seq>/*.jpg 的根目錄")
    ap.add_argument("--sample", type=Path,
                    help="官方 sample_submission.csv；給了就當契約驗（序列集合＋逐序列幀數）")
    ap.add_argument("--init-dir", type=Path,
                    help="init 另外發佈時的目錄（找 <seq>.txt / <seq>/init_rect.txt）")
    ap.add_argument("--prefix", default=None,
                    help="序列名前綴（nir-/rednir-/vis-）。預設依官方模態資料夾名自動推；"
                         "來源不是官方佈局又要加前綴時明示；傳空字串＝明示不加")
    ap.add_argument("--write-sample", type=Path, default=None,
                    help="沒有官方 sample_submission.csv 時，從驗證通過的序列產一份"
                         "（ID,x,y,width,height；<seq>_<1..n>）；與 --sample 互斥")
    ap.add_argument("--renumber", action="store_true",
                    help="允許把非數字幀名依自然序重編為 0001..N（映射寫入 manifest）")
    ap.add_argument("--dry-run", action="store_true", help="只檢查與報告，不寫任何檔案")
    a = ap.parse_args()

    if a.sample is not None and a.write_sample is not None:
        raise SystemExit("FATAL: --sample 與 --write-sample 互斥（有官方 sample 就用官方的）")

    sources = discover(a.source, a.prefix)
    print(f"[discover] 找到 {len(sources)} 支序列（來源：{a.source}）")
    by_prefix: dict[str, int] = {}
    for s in sources:
        p = s.name.split("-", 1)[0] + "-" if s.name.split("-", 1)[0] in ("nir", "rednir", "vis") else "(無前綴)"
        by_prefix[p] = by_prefix.get(p, 0) + 1
    print(f"[discover] 模態前綴分佈：{by_prefix}")
    if "(無前綴)" in by_prefix and a.prefix is None:
        print("  ⚠️ 有序列沒有模態前綴：管線會把它們全部當 VIS（RedNIR 品質頭不觸發、"
              "one-hot 特徵錯）。若來源不是官方佈局，請用 --prefix 明示。", file=sys.stderr)

    sample_counts = load_sample_counts(a.sample) if a.sample else None
    if sample_counts is not None:
        print(f"[sample] {len(sample_counts)} 支序列、{sum(sample_counts.values())} 列")

    manifest: dict[str, dict] = {}
    problems: list[str] = []

    for src in sorted(sources, key=lambda s: s.name):
        seq = src.name
        names = src.frames()
        if not names:
            problems.append(f"{seq}: 無影像")
            continue

        # ── 幀名：stem 必須可 int()，否則預設拒絕 ──────────────────────────
        non_numeric = [n for n in names if not Path(n).stem.isdigit()]
        renamed: dict[str, str] | None = None
        if non_numeric:
            if not a.renumber:
                problems.append(
                    f"{seq}: {len(non_numeric)} 個幀名 stem 非純數字（例 {non_numeric[:2]}）；"
                    f"確認排序意圖後加 --renumber")
                continue
            width = max(4, len(str(len(names))))
            renamed = {n: f"{i:0{width}d}{Path(n).suffix.lower()}"
                       for i, n in enumerate(names, start=1)}

        out_names = [renamed[n] if renamed else n for n in names]
        if len(set(out_names)) != len(out_names):
            problems.append(f"{seq}: 輸出檔名重複")
            continue

        # ── init 框：zip/目錄內 → --init-dir → 皆無即 BLOCK ────────────────
        init_text = src.init_text()
        init_origin = "in-source"
        if init_text is None and a.init_dir:
            for cand in (a.init_dir / f"{seq}.txt",
                         a.init_dir / seq / "init_rect.txt",
                         a.init_dir / seq / "groundtruth_rect.txt"):
                if cand.is_file():
                    init_text = cand.read_text(encoding="utf-8-sig", errors="replace")
                    init_origin = f"init-dir:{cand.name}"
                    break
        if init_text is None:
            problems.append(f"{seq}: 找不到 init_rect/groundtruth（--init-dir 也沒有）")
            continue
        box = parse_box(init_text)
        if box is None:
            problems.append(f"{seq}: init 內容解析不出四個數：{init_text[:60]!r}")
            continue
        x, y, w, h = box
        if w <= 0 or h <= 0:
            problems.append(f"{seq}: init 寬高非正 ({x:g},{y:g},{w:g},{h:g})"
                            f"——可能是 (x1,y1,x2,y2) 慣例")
            continue

        # ── 座標慣例：唯一會燒掉整輪推論的失敗型態 ─────────────────────────
        first = src.read(names[0])
        if png_bit_depth(first) == 16:
            problems.append(f"{seq}: 首幀是 16-bit PNG＝mosaic 原始資料，管線只吃假色 jpg；"
                            "請改指 *-Falsecolor 資料夾")
            continue
        size = image_size(first)
        if size is None:
            problems.append(f"{seq}: 無法判定影像尺寸（缺 PIL 且非標準 JPEG/PNG 檔頭）")
            continue
        iw, ih = size
        if x < -BOUND_TOL or y < -BOUND_TOL or x + w > iw + BOUND_TOL or y + h > ih + BOUND_TOL:
            problems.append(
                f"{seq}: init ({x:g},{y:g},{w:g},{h:g}) 超出影像 {iw}x{ih}"
                f"——極可能是 (x1,y1,x2,y2) 被當成 (x,y,w,h)")
            continue

        if sample_counts is not None:
            if seq not in sample_counts:
                problems.append(f"{seq}: 不在 sample 的序列集合內")
                continue
            if sample_counts[seq] != len(names):
                problems.append(f"{seq}: 幀數 {len(names)} 與 sample 的 {sample_counts[seq]} 不符")
                continue

        manifest[seq] = {
            "n_frames": len(names),
            "image_size": [iw, ih],
            "init_rect": [x, y, w, h],
            "init_origin": init_origin,
            "origin": src.origin,
            "renumbered": bool(renamed),
            "first_frame_out": out_names[0],
            "last_frame_out": out_names[-1],
        }
        if renamed:
            manifest[seq]["rename_map_sample"] = dict(list(renamed.items())[:3])

        if a.dry_run:
            continue

        # ── 原子寫入：先 .part 再 os.replace；既有完整目錄則跳過、不完整則拒絕覆寫 ──
        dst = a.frames_out / seq
        if dst.is_dir():
            have = sorted((p.name for p in dst.iterdir()
                           if p.is_file() and p.suffix.lower() in IMG_EXTS), key=natural_key)
            if have == sorted(out_names, key=natural_key) and (dst / "init_rect.txt").is_file():
                manifest[seq]["skipped_existing"] = True
                continue
            problems.append(f"{seq}: 目的地已存在且不相符；拒絕覆寫，請人工隔離")
            continue
        part = dst.with_name(dst.name + ".part")
        if part.exists():
            problems.append(f"{seq}: 發現上次中斷的 {part.name}；拒絕自動刪除")
            continue
        part.mkdir(parents=True)
        for src_name, out_name in zip(names, out_names):
            (part / out_name).write_bytes(src.read(src_name))
        (part / "init_rect.txt").write_text(
            " ".join(format(v, ".15g") for v in (x, y, w, h)) + "\n")
        os.replace(part, dst)

    # ── 契約：sample 有而來源沒有的序列 ────────────────────────────────────
    if sample_counts is not None:
        missing = sorted(set(sample_counts) - set(manifest))
        if missing:
            problems.append(f"sample 有但未產出的序列 {len(missing)} 支：{missing[:5]}")

    n_ok = len(manifest)
    total_frames = sum(v["n_frames"] for v in manifest.values())
    print(f"[result] 成功 {n_ok} 支／{total_frames} 幀；問題 {len(problems)} 項")
    for p in problems:
        print(f"  [BLOCK] {p}", file=sys.stderr)

    if not a.dry_run and manifest:
        a.frames_out.mkdir(parents=True, exist_ok=True)
        (a.frames_out / "INGEST_MANIFEST.json").write_text(
            json.dumps({"n_sequences": n_ok, "n_frames": total_frames,
                        "source": str(a.source), "sequences": manifest},
                       indent=1, ensure_ascii=False))
        print(f"[manifest] {a.frames_out / 'INGEST_MANIFEST.json'}")

    if a.write_sample is not None and manifest and not problems:
        if a.dry_run:
            print(f"[sample] dry-run：會寫 {total_frames} 列到 {a.write_sample}（未寫）")
        else:
            a.write_sample.parent.mkdir(parents=True, exist_ok=True)
            n_rows = write_sample_csv(a.write_sample, manifest)
            print(f"[sample] 已寫 {a.write_sample}：{n_ok} 支／{n_rows} 列")

    if problems:
        print("\n⚠️ 有 BLOCK：**不要**繼續跑 run_ranking_b.py。先解決上面每一項。", file=sys.stderr)
        return 2
    print("\n✅ ingestion 契約全過。下一步：run_ranking_b.py 的 dry-run（必須 BLOCK=0）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
