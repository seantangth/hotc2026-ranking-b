#!/usr/bin/env python3
"""跨執行共識（run-ensemble medoid）——D096。

背景：08-30 實測同一條鏈的三次執行 LB 為 0.71666 / 0.71246 / 0.70701（全距 0.00965），
大於當時剩餘所有加分手段的總和。Ranking B 只跑一次 ⇒ 那一次是抽籤。本工具把
多次執行的輸出合成一份，讓「某一次跑飛」不會單獨決定結果。

規則：逐幀取 **medoid** ＝ 與其他各框 IoU 總和最大的那一個框。
- 不合成新框（不取座標中位數／平均）：E33 已判死 mask→box 導出慣例整族，
  9 個變體全負，合成框比真實 tight box 差。medoid 只從實際輸出裡挑一個。
- 平手取第一個輸入（＝現任 / incumbent），避免無謂漂移。
- 某次執行追到別的物體時，它與其他框的 IoU 皆 ~0 ⇒ 總和最小 ⇒ 自動出局；
  規則因此在 N=3 時退化為多數決。

⚠️ N=2 時 medoid 恆等於現任（sum 對兩者都是同一個 IoU 值，平手 → 取第一個）
   ⇒ **兩次執行測不到任何東西**。08-30 的 v084 有 40 支序列正是這個情形
   （只有兩份獨立答案），該批序列等於沒被驗證。用前務必確認每支序列都有 ≥3 份。
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

Box = tuple[float, float, float, float]

HEADER = ["ID", "x", "y", "width", "height"]


def _fmt(v: float) -> str:
    """整數值印成整數、其餘保留原樣——最終 submission 是整數格式，
    但管線中間產物（crop_rerun merge 的輸出）是浮點（如 `33.0`）。
    ⚠️ 08-31 實測：medoid 起初只用整數版最終檔測過，餵中間產物時 int(v) 直接炸。"""
    return str(int(v)) if float(v).is_integer() else repr(v)


def load(path: Path) -> dict[str, Box]:
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header != HEADER:
            raise SystemExit(f"表頭不符 {path}: {header}")
        rows: dict[str, Box] = {}
        for row in reader:
            if row[0] in rows:
                raise SystemExit(f"重複 ID {row[0]} in {path}")
            try:
                rows[row[0]] = tuple(float(v) for v in row[1:])  # type: ignore[assignment]
            except ValueError as exc:
                raise SystemExit(f"{path} 的 {row[0]} 含非數值：{row[1:]}") from exc
    return rows


def iou(a: Box, b: Box) -> float:
    if a == b:
        return 1.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter)


def medoid(boxes: list[Box]) -> tuple[Box, int]:
    """回傳 (勝出的框, 它在 boxes 裡的索引)。平手取索引最小者（＝現任）。"""
    if len(boxes) < 2:
        return boxes[0], 0
    best_idx, best_score = 0, None
    for i, bi in enumerate(boxes):
        score = sum(iou(bi, bj) for j, bj in enumerate(boxes) if j != i)
        if best_score is None or score > best_score:
            best_idx, best_score = i, score
    return boxes[best_idx], best_idx


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path,
                    help="各次執行的 submission CSV；**第一個視為現任**（平手時勝出）")
    ap.add_argument("--out", type=Path, required=True, help="輸出的共識 CSV")
    ap.add_argument("--report", type=Path, help="統計報告輸出路徑（預設印到 stdout）")
    args = ap.parse_args(argv)

    if len(args.runs) < 2:
        raise SystemExit("至少要兩份執行結果")
    if len(args.runs) == 2:
        print("⚠️ 只有兩份執行：medoid 恆等於現任，本次不會有任何改變（見 docstring）",
              file=sys.stderr)

    tables = [load(p) for p in args.runs]
    keys = set(tables[0])
    for path, tbl in zip(args.runs[1:], tables[1:]):
        if set(tbl) != keys:
            raise SystemExit(f"ID 集合不一致：{path}")

    incumbent = tables[0]
    out: dict[str, Box] = {}
    picked = Counter()
    changed_by_seq = Counter()
    all_same = 0
    for key in incumbent:
        boxes = [t[key] for t in tables]
        if all(b == boxes[0] for b in boxes[1:]):
            out[key] = boxes[0]
            all_same += 1
            picked["all_same"] += 1
            continue
        box, idx = medoid(boxes)
        out[key] = box
        picked[args.runs[idx].name] += 1
        if box != incumbent[key]:
            changed_by_seq[key.rsplit("_", 1)[0]] += 1

    # hard checks：首幀不得被改動（官方 init box 必須原樣保留）
    first_frames = [k for k in out if k.rsplit("_", 1)[1] == "1"]
    moved = [k for k in first_frames if out[k] != incumbent[k]]
    if moved:
        raise SystemExit(f"首幀被改動（應為官方 init box）：{moved[:5]}")
    if any(w <= 0 or h <= 0 for _, _, w, h in out.values()):
        raise SystemExit("輸出含非正的寬或高")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        for key in incumbent:  # 保持與現任相同的列順序（canonical order）
            writer.writerow([key, *(_fmt(v) for v in out[key])])

    changed = sum(changed_by_seq.values())
    lines = [
        f"輸入 {len(tables)} 份執行：" + ", ".join(p.name for p in args.runs),
        f"總列數 {len(out)}；首幀 {len(first_frames)} 支全部保留 init box",
        f"三者全同 {all_same}（{all_same / len(out):.1%}）",
        "medoid 取用來源：" + ", ".join(
            f"{k}={v}" for k, v in picked.most_common() if k != "all_same"),
        f"相對現任變動 {changed} 列（{changed / len(out):.2%}）／涉及 {len(changed_by_seq)} 支序列",
        "變動最多的 10 支：" + ", ".join(
            f"{s}:{n}" for s, n in changed_by_seq.most_common(10)),
    ]
    text = "\n".join(lines)
    if args.report:
        args.report.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
