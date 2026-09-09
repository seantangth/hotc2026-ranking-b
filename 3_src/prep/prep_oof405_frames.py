#!/usr/bin/env python3
"""建立 train405 zero-shot prediction cache 的影像與不可含糊 frame contract。

輸入是 gDrive ``raw_archive`` 的訓練假色 zip 與 ``2026training.csv``。輸出：

* ``frames/<modality-name>/*.jpg``：兩個 frozen tracker 共用的 405 支影片；
* ``contracts/sample_contract.csv``：只含實際存在影像所對應的官方全域 ID；
* ``contracts/observed_gt.csv`` / ``excluded_gt.csv``：實體幀 GT 與隔離列；
* ``contracts/mirror_map.csv`` / ``scoring_gt.csv``：完全重複 GT block 的官方列敏感度；
* ``contracts/frame_contract.csv`` / ``capture_groups.csv`` / ``folds.json``。

已知 basketball1/2、pills5、rubik 的 GT 有多個不連續區塊且 zip 較短。本腳本只接受
「區塊長度等於實際影像數」的明確映射；若多個候選不是逐框完全相同就 fail closed，
絕不以最後一框補滿不存在的影像。physical-capture primary group 使用去 modality 後的
完整 basename；不做會把 car1..car86 全併在一起的 strip-digits family grouping。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


SUB_COLS = ["ID", "x", "y", "width", "height"]
MOD_FOLDERS = {
    "nir": ("HSI-NIR-FalseColor",),
    "rednir": ("HSI-RedNIR-FalseColor",),
    "vis": ("HSI-VIS-FalseColor", "HSI-VIS-FalseColor_25"),
}
IMG_EXTS = {".jpg", ".jpeg"}


@dataclass(frozen=True)
class GtRow:
    ident: str
    seq: str
    frame: int
    x: float
    y: float
    width: float
    height: float

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.width, self.height

    @property
    def valid_box(self) -> bool:
        return all(math.isfinite(v) for v in self.box) and self.width > 0 and self.height > 0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_gt(path: Path) -> dict[str, list[GtRow]]:
    groups: dict[str, list[GtRow]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != SUB_COLS:
            raise ValueError(f"GT 欄位 {reader.fieldnames} != {SUB_COLS}")
        seen: set[str] = set()
        for line, row in enumerate(reader, 2):
            ident = row["ID"]
            if ident in seen:
                raise ValueError(f"GT ID 重複 line {line}: {ident}")
            seen.add(ident)
            try:
                seq, frame_s = ident.rsplit("_", 1)
                item = GtRow(
                    ident, seq, int(frame_s),
                    float(row["x"]), float(row["y"]),
                    float(row["width"]), float(row["height"]),
                )
            except (ValueError, KeyError) as exc:
                raise ValueError(f"GT line {line} 無效：{row}") from exc
            groups[seq].append(item)
    for seq, rows in groups.items():
        rows.sort(key=lambda r: r.frame)
        if not re.match(r"^(nir|rednir|vis)-.+$", seq):
            raise ValueError(f"未知序列命名：{seq}")
    return dict(groups)


def contiguous_blocks(rows: list[GtRow]) -> list[list[GtRow]]:
    blocks: list[list[GtRow]] = []
    for row in rows:
        if not blocks or row.frame != blocks[-1][-1].frame + 1:
            blocks.append([row])
        else:
            blocks[-1].append(row)
    return blocks


def boxes_identical(a: list[GtRow], b: list[GtRow]) -> bool:
    return len(a) == len(b) and all(x.box == y.box for x, y in zip(a, b, strict=True))


# Max per-coordinate archive-vs-GT init disagreement tolerated when the GT block
# was not chosen but taken whole (see choose_observed_block).  1px: annotation
# drift.  Anything larger is a real disagreement and must still fail closed.
INIT_ANNOTATION_TOL = 1.0


def box_matches(a: tuple[float, ...], b: tuple[float, ...], tol: float = 1e-6) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


def choose_observed_block(
    seq: str, rows: list[GtRow], n_images: int, archive_init: tuple[float, ...] | None,
) -> tuple[list[GtRow], str, tuple[float, ...] | None]:
    blocks = contiguous_blocks(rows)
    if len(rows) == n_images:
        candidates = [rows]
        reason = "all_gt_rows"
    else:
        candidates = [block for block in blocks if len(block) == n_images]
        reason = "matching_contiguous_block"
    if archive_init is not None:
        init_matches = [b for b in candidates if box_matches(b[0].box, archive_init)]
        if init_matches:
            candidates = init_matches
    if not candidates:
        summary = [(b[0].frame, b[-1].frame, len(b)) for b in blocks]
        raise ValueError(
            f"{seq}: 無法將 {n_images} 張影像映射到 GT blocks={summary}, init={archive_init}")
    if len(candidates) > 1:
        first = candidates[0]
        if not all(boxes_identical(first, other) for other in candidates[1:]):
            summary = [(b[0].frame, b[-1].frame) for b in candidates]
            raise ValueError(f"{seq}: 有多個非同值候選 GT blocks {summary}，拒絕猜測")
        reason = "identical_duplicate_block_first_canonical"
    chosen = candidates[0]
    archive_init_delta: tuple[float, ...] | None = None
    if archive_init is not None and not box_matches(chosen[0].box, archive_init):
        # The archive init exists to pin *frame alignment*: it tells us which GT
        # block the images correspond to.  When the whole GT was taken verbatim
        # (one block, len(rows) == n_images) there was no mapping to choose --
        # position 0 is GT row 0 by construction -- so a residual disagreement is
        # annotation drift between two files describing the same frame, not
        # misalignment.  Known cases: nir-rider15 (x,w) and vis-cup3 (w), both 1px.
        # Anywhere a block actually had to be picked, this stays exact: loosening
        # it there could silently select the wrong block.
        unambiguous = reason == "all_gt_rows" and len(blocks) == 1
        delta = tuple(a - b for a, b in zip(archive_init, chosen[0].box, strict=True))
        if not (unambiguous and all(abs(d) <= INIT_ANNOTATION_TOL for d in delta)):
            raise ValueError(
                f"{seq}: archive init {archive_init} != chosen GT first {chosen[0].box}")
        archive_init_delta = delta
    return chosen, reason, archive_init_delta


def pick_zip(root: Path, seq: str) -> Path:
    modality, name = seq.split("-", 1)
    tried: list[Path] = []
    for folder in MOD_FOLDERS[modality]:
        for rel in (
            Path("update") / folder / f"{name}.zip",
            Path("training") / "update" / folder / f"{name}.zip",
            Path("training") / folder / f"{name}.zip",
        ):
            candidate = root / rel
            tried.append(candidate)
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"{seq}: 找不到假色 zip；已試 {[str(p) for p in tried]}")


def archive_images(zf: zipfile.ZipFile, seq: str) -> list[zipfile.ZipInfo]:
    members = [m for m in zf.infolist() if not m.is_dir() and Path(m.filename).suffix.lower() in IMG_EXTS]
    if not members:
        raise ValueError(f"{seq}: zip 內無 jpg/jpeg")
    by_name: dict[str, zipfile.ZipInfo] = {}
    for member in members:
        name = Path(member.filename).name
        if not Path(name).stem.isdigit():
            raise ValueError(f"{seq}: 非純數字 frame 名：{member.filename}")
        if name in by_name:
            raise ValueError(f"{seq}: zip 內重複 basename：{name}")
        by_name[name] = member
    return sorted(by_name.values(), key=lambda m: int(Path(m.filename).stem))


def archive_init_box(zf: zipfile.ZipFile) -> tuple[float, float, float, float] | None:
    preferred = []
    for member in zf.infolist():
        base = Path(member.filename).name.lower()
        if not member.is_dir() and base in {"init_rect.txt", "groundtruth_rect.txt"}:
            preferred.append(member)
    for member in sorted(preferred, key=lambda m: ("init" not in Path(m.filename).name.lower(), m.filename)):
        text = zf.read(member).decode("utf-8-sig", errors="strict").splitlines()
        if not text:
            continue
        tokens = [x for x in re.split(r"[,\s]+", text[0].strip()) if x]
        if len(tokens) >= 4:
            return tuple(float(x) for x in tokens[:4])  # type: ignore[return-value]
    return None


def materialize_sequence(zf: zipfile.ZipFile, members: list[zipfile.ZipInfo],
                         dst: Path, init: tuple[float, ...]) -> None:
    expected_names = [Path(m.filename).name for m in members]
    if dst.is_dir():
        existing = sorted(
            [p.name for p in dst.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
            key=lambda n: int(Path(n).stem),
        )
        if existing == expected_names and (dst / "init_rect.txt").is_file():
            return
        raise RuntimeError(f"{dst}: 既有資料不完整/不相符；拒絕覆寫，請人工隔離後重跑")
    part = dst.with_name(dst.name + ".part")
    if part.exists():
        raise RuntimeError(f"{part}: 發現上次中斷的 part 目錄；拒絕自動刪除")
    part.mkdir(parents=True)
    try:
        for member in members:
            target = part / Path(member.filename).name
            with zf.open(member) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
        (part / "init_rect.txt").write_text(" ".join(format(v, ".15g") for v in init) + "\n")
        os.replace(part, dst)
    except Exception:
        # 保留 .part 作事故證據；下一次會 fail closed，不會誤當完成品。
        raise


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    with part.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(part, path)


def assign_primary_folds(observed: dict[str, list[GtRow]], n_folds: int = 5) -> tuple[dict[str, int], dict]:
    group_to_seqs: dict[str, list[str]] = defaultdict(list)
    for seq in observed:
        group_to_seqs[seq.split("-", 1)[1]].append(seq)
    group_weight = {g: sum(len(observed[s]) for s in seqs) for g, seqs in group_to_seqs.items()}
    fold_frames = [0] * n_folds
    fold_seqs = [0] * n_folds
    group_fold: dict[str, int] = {}
    for group in sorted(group_to_seqs, key=lambda g: (-group_weight[g], g)):
        fold = min(range(n_folds), key=lambda f: (fold_frames[f], fold_seqs[f], f))
        group_fold[group] = fold
        fold_frames[fold] += group_weight[group]
        fold_seqs[fold] += len(group_to_seqs[group])
    seq_fold = {seq: group_fold[group] for group, seqs in group_to_seqs.items() for seq in seqs}
    meta = {
        "schema_version": 1,
        "group_rule": "strip modality prefix only; preserve full basename",
        "n_folds": n_folds,
        "n_groups": len(group_to_seqs),
        "n_cross_modality_groups": sum(len(v) > 1 for v in group_to_seqs.values()),
        "fold_frames": fold_frames,
        "fold_sequences": fold_seqs,
        "group_to_fold": dict(sorted(group_fold.items())),
    }
    return seq_fold, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-root", required=True, help="含 training/ 與 update/ 的 raw_archive 根")
    parser.add_argument("--gt", required=True, help="2026training.csv")
    parser.add_argument("--frames-out", required=True, help="輸出 <seq>/*.jpg 根目錄")
    parser.add_argument("--contracts-out", required=True, help="輸出 contracts 目錄")
    parser.add_argument("--no-extract", action="store_true", help="只驗 zip/產 contract，不解影像")
    parser.add_argument("--hash-zips", action="store_true", help="另算每個來源 zip SHA256（正式 run 必開）")
    args = parser.parse_args()

    zip_root = Path(args.zip_root).expanduser().resolve()
    gt_path = Path(args.gt).expanduser().resolve()
    frames_out = Path(args.frames_out).expanduser().resolve()
    contracts_out = Path(args.contracts_out).expanduser().resolve()
    gt = load_gt(gt_path)
    if len(gt) != 405:
        raise SystemExit(f"GT 應為 405 序列，實得 {len(gt)}")

    observed: dict[str, list[GtRow]] = {}
    selections: dict[str, dict] = {}
    zip_hashes: dict[str, str] = {}
    for index, seq in enumerate(sorted(gt), 1):
        archive = pick_zip(zip_root, seq)
        with zipfile.ZipFile(archive) as zf:
            bad = zf.testzip()
            if bad:
                raise ValueError(f"{seq}: zip CRC 失敗 member={bad}")
            members = archive_images(zf, seq)
            init_from_archive = archive_init_box(zf)
            chosen, reason, init_delta = choose_observed_block(
                seq, gt[seq], len(members), init_from_archive)
            init = chosen[0].box
            if not args.no_extract:
                frames_out.mkdir(parents=True, exist_ok=True)
                materialize_sequence(zf, members, frames_out / seq, init)
        observed[seq] = chosen
        selections[seq] = {
            "source_zip": str(archive.relative_to(zip_root)),
            "n_images": len(members),
            "n_gt_total": len(gt[seq]),
            "selected_start": chosen[0].frame,
            "selected_end": chosen[-1].frame,
            "selection_reason": reason,
            "archive_init_delta": list(init_delta) if init_delta else None,
            "archive_init": list(init_from_archive) if init_from_archive is not None else None,
        }
        if args.hash_zips:
            zip_hashes[str(archive.relative_to(zip_root))] = sha256_file(archive)
        if index % 25 == 0 or index == len(gt):
            print(f"[{index}/{len(gt)}] contract/extract OK", flush=True)

    seq_fold, fold_meta = assign_primary_folds(observed)
    selected_ids = {r.ident for rows in observed.values() for r in rows}
    sample_rows, observed_rows, excluded_rows, frame_rows = [], [], [], []
    mirror_rows, mirror_map_rows, unobservable_rows = [], [], []
    capture_rows = []
    for seq in sorted(observed):
        modality, group = seq.split("-", 1)
        selection = selections[seq]
        capture_rows.append({
            "sequence": seq, "modality": modality, "capture_group": group,
            "fold": seq_fold[seq], "n_frames": len(observed[seq]),
            "evidence": "exact basename after modality removal",
        })
        for position, row in enumerate(observed[seq]):
            sample_rows.append({"ID": row.ident, "x": 0, "y": 0, "width": 0, "height": 0})
            observed_rows.append({
                "ID": row.ident, "x": row.x, "y": row.y,
                "width": row.width, "height": row.height,
            })
            frame_rows.append({
                "ID": row.ident, "sequence": seq, "position": position,
                "modality": modality, "capture_group": group, "fold": seq_fold[seq],
                "gt_valid": int(row.valid_box),
                "selected_block_start": selection["selected_start"],
                "selected_block_end": selection["selected_end"],
            })
    for seq in sorted(gt):
        chosen = observed[seq]
        for block in contiguous_blocks(gt[seq]):
            if all(row.ident in selected_ids for row in block):
                continue
            mirrorable = boxes_identical(chosen, block)
            for position, row in enumerate(block):
                reason = ("identical_duplicate_gt_block_mirrorable" if mirrorable
                          else "no_corresponding_image_in_official_zip")
                excluded_rows.append({
                    "ID": row.ident, "x": row.x, "y": row.y,
                    "width": row.width, "height": row.height, "reason": reason,
                })
                if mirrorable:
                    source = chosen[position]
                    mirror_rows.append({
                        "ID": row.ident, "x": row.x, "y": row.y,
                        "width": row.width, "height": row.height,
                    })
                    mirror_map_rows.append({
                        "source_ID": source.ident, "target_ID": row.ident,
                        "sequence": seq, "position": position,
                    })
                else:
                    unobservable_rows.append({
                        "ID": row.ident, "x": row.x, "y": row.y,
                        "width": row.width, "height": row.height,
                        "reason": reason,
                    })

    write_csv_atomic(contracts_out / "sample_contract.csv", SUB_COLS, sample_rows)
    write_csv_atomic(contracts_out / "observed_gt.csv", SUB_COLS, observed_rows)
    write_csv_atomic(
        contracts_out / "excluded_gt.csv", SUB_COLS + ["reason"], excluded_rows)
    write_csv_atomic(
        contracts_out / "unobservable_gt.csv", SUB_COLS + ["reason"], unobservable_rows)
    write_csv_atomic(
        contracts_out / "mirror_map.csv",
        ["source_ID", "target_ID", "sequence", "position"], mirror_map_rows)
    scoring_rows = sorted(observed_rows + mirror_rows, key=lambda r: (r["ID"].rsplit("_", 1)[0],
                                                                       int(r["ID"].rsplit("_", 1)[1])))
    write_csv_atomic(contracts_out / "scoring_gt.csv", SUB_COLS, scoring_rows)
    frame_fields = [
        "ID", "sequence", "position", "modality", "capture_group", "fold", "gt_valid",
        "selected_block_start", "selected_block_end",
    ]
    write_csv_atomic(contracts_out / "frame_contract.csv", frame_fields, frame_rows)
    capture_fields = ["sequence", "modality", "capture_group", "fold", "n_frames", "evidence"]
    write_csv_atomic(contracts_out / "capture_groups.csv", capture_fields, capture_rows)

    contracts_out.mkdir(parents=True, exist_ok=True)
    folds_path = contracts_out / "folds.json"
    folds_doc = {**fold_meta, "sequence_to_fold": dict(sorted(seq_fold.items()))}
    folds_path.write_text(json.dumps(folds_doc, indent=2, ensure_ascii=False) + "\n")
    report = {
        "schema_version": 1,
        "gt_csv": str(gt_path),
        "gt_sha256": sha256_file(gt_path),
        "sequences": len(gt),
        "gt_rows_total": sum(len(v) for v in gt.values()),
        "physical_prediction_rows": len(observed_rows),
        "mirrorable_duplicate_gt_rows": len(mirror_rows),
        "scoring_rows_with_mirror_sensitivity": len(scoring_rows),
        "excluded_rows_total": len(excluded_rows),
        "unobservable_gt_rows": len(unobservable_rows),
        "invalid_observed_gt_boxes": sum(not r.valid_box for rows in observed.values() for r in rows),
        "selection": selections,
        "folds": fold_meta,
        "zip_sha256": zip_hashes,
    }
    report_path = contracts_out / "prep_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    hashes = {}
    for path in sorted(contracts_out.iterdir()):
        if path.is_file() and path.name != "contract_sha256.json":
            hashes[path.name] = sha256_file(path)
    (contracts_out / "contract_sha256.json").write_text(
        json.dumps(hashes, indent=2, ensure_ascii=False) + "\n")
    print(
        f"OK: {len(gt)} seq / {len(observed_rows)} observed / {len(excluded_rows)} excluded / "
        f"{fold_meta['n_groups']} capture groups → {contracts_out}")


if __name__ == "__main__":
    main()
