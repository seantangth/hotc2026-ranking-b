"""prep_rankingb_frames_v1 的回歸測試 — 把 09-06 手測過的 7 個情境固化。

9/7 的 ingestion 只跑一次、且在三天硬窗口內，所以「壞輸入必須被擋下」這件事
不能只靠當天目視。這裡用手工合成的最小 PNG（不需 PIL）建出各種來源形狀。
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "prep_rankingb_frames_v1.py"


def make_png(w: int, h: int) -> bytes:
    """最小合法 PNG（灰階 8-bit 全黑），供 image_size 的檔頭解析路徑使用。"""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x00" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def build_seq_dir(root: Path, seq: str, n: int = 5, *, size=(64, 48),
                  init: str = "10 10 20 15", names=None) -> Path:
    d = root / seq
    d.mkdir(parents=True)
    png = make_png(*size)
    for i in range(1, n + 1):
        name = names[i - 1] if names else f"{i:04d}.png"
        (d / name).write_bytes(png)
    (d / "init_rect.txt").write_text(init + "\n")
    return d


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True)


def write_sample(path: Path, counts: dict[str, int]) -> Path:
    lines = ["ID,x,y,width,height"]
    for seq, n in counts.items():
        lines += [f"{seq}_{i},0,0,0,0" for i in range(1, n + 1)]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_dir_source_roundtrip(tmp_path):
    """已解開的目錄 → 輸出契約（數字幀名 ＋ 空白分隔 init_rect ＋ manifest）。"""
    src = tmp_path / "src"
    build_seq_dir(src, "nir-a", n=5)
    build_seq_dir(src, "vis-b", n=3)
    out = tmp_path / "out"
    r = run("--source", src, "--frames-out", out)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ["nir-a", "vis-b"]
    assert (out / "nir-a" / "init_rect.txt").read_text().strip() == "10 10 20 15"
    assert sorted(p.name for p in (out / "vis-b").glob("*.png")) == [
        "0001.png", "0002.png", "0003.png"]
    man = json.loads((out / "INGEST_MANIFEST.json").read_text())
    assert man["n_sequences"] == 2 and man["n_frames"] == 8
    assert man["sequences"]["nir-a"]["image_size"] == [64, 48]


def test_zip_per_sequence(tmp_path):
    """每序列一個 zip（最可能的官方形狀）。"""
    stage = tmp_path / "stage"
    build_seq_dir(stage, "nir-a", n=4)
    zdir = tmp_path / "zips"
    zdir.mkdir()
    with zipfile.ZipFile(zdir / "nir-a.zip", "w") as zf:
        for p in sorted((stage / "nir-a").iterdir()):
            zf.write(p, f"nir-a/{p.name}")
    out = tmp_path / "out"
    assert run("--source", zdir, "--frames-out", out).returncode == 0
    assert len(list((out / "nir-a").glob("*.png"))) == 4
    assert (out / "nir-a" / "init_rect.txt").is_file()


def test_single_big_zip_with_sequence_dirs(tmp_path):
    """單一大 zip 內含多個序列目錄。"""
    stage = tmp_path / "stage"
    build_seq_dir(stage, "nir-a", n=3)
    build_seq_dir(stage, "rednir-c", n=2)
    big = tmp_path / "all.zip"
    with zipfile.ZipFile(big, "w") as zf:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                zf.write(p, str(p.relative_to(stage)))
    out = tmp_path / "out"
    assert run("--source", big, "--frames-out", out).returncode == 0
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ["nir-a", "rednir-c"]


def test_x1y1x2y2_convention_is_blocked(tmp_path):
    """**唯一會燒掉整輪 4.5 小時的失敗型態**：init 被寫成 (x1,y1,x2,y2)。"""
    src = tmp_path / "src"
    # 影像 64x48；(10,10,60,40) 當 xywh 解讀 ⇒ x+w=70 > 64 ⇒ 必須 BLOCK
    build_seq_dir(src, "nir-a", size=(64, 48), init="10 10 60 40")
    out = tmp_path / "out"
    r = run("--source", src, "--frames-out", out)
    assert r.returncode == 2
    assert "超出影像" in r.stderr and "x1,y1,x2,y2" in r.stderr
    assert not (out / "nir-a").exists(), "BLOCK 時不得留下半成品目錄"


def test_nonpositive_wh_is_blocked(tmp_path):
    src = tmp_path / "src"
    build_seq_dir(src, "nir-a", init="10 10 0 15")
    r = run("--source", src, "--frames-out", tmp_path / "out")
    assert r.returncode == 2 and "寬高非正" in r.stderr


def test_non_numeric_stem_blocked_then_renumber(tmp_path):
    """非數字幀名預設拒絕（會讓 track_t1 的排序對應靜默錯位），--renumber 才放行。"""
    src = tmp_path / "src"
    build_seq_dir(src, "nir-a", n=3, names=["frame_a.png", "frame_b.png", "frame_c.png"])
    out = tmp_path / "out"
    r = run("--source", src, "--frames-out", out)
    assert r.returncode == 2 and "非純數字" in r.stderr

    out2 = tmp_path / "out2"
    r2 = run("--source", src, "--frames-out", out2, "--renumber")
    assert r2.returncode == 0, r2.stderr
    assert sorted(p.name for p in (out2 / "nir-a").glob("*.png")) == [
        "0001.png", "0002.png", "0003.png"]
    man = json.loads((out2 / "INGEST_MANIFEST.json").read_text())
    assert man["sequences"]["nir-a"]["renumbered"] is True


def test_missing_init_is_blocked(tmp_path):
    src = tmp_path / "src"
    d = build_seq_dir(src, "nir-a")
    (d / "init_rect.txt").unlink()
    r = run("--source", src, "--frames-out", tmp_path / "out")
    assert r.returncode == 2 and "找不到 init" in r.stderr


def test_init_dir_fallback(tmp_path):
    """init 另外發佈時，--init-dir 接得起來。"""
    src = tmp_path / "src"
    d = build_seq_dir(src, "nir-a")
    (d / "init_rect.txt").unlink()
    init_dir = tmp_path / "inits"
    init_dir.mkdir()
    (init_dir / "nir-a.txt").write_text("1,2,3,4\n")  # 逗號分隔亦須吃得下
    out = tmp_path / "out"
    assert run("--source", src, "--frames-out", out, "--init-dir", init_dir).returncode == 0
    assert (out / "nir-a" / "init_rect.txt").read_text().strip() == "1 2 3 4"


@pytest.mark.parametrize("bad", ["count", "missing_seq"])
def test_sample_contract_violations_blocked(tmp_path, bad):
    src = tmp_path / "src"
    build_seq_dir(src, "nir-a", n=5)
    counts = {"nir-a": 4} if bad == "count" else {"nir-a": 5, "vis-ghost": 3}
    sample = write_sample(tmp_path / "sample.csv", counts)
    r = run("--source", src, "--frames-out", tmp_path / "out", "--sample", sample, "--dry-run")
    assert r.returncode == 2
    assert ("幀數 5 與 sample 的 4 不符" in r.stderr) or ("vis-ghost" in r.stderr)


def test_sample_contract_pass_and_dry_run_writes_nothing(tmp_path):
    src = tmp_path / "src"
    build_seq_dir(src, "nir-a", n=5)
    sample = write_sample(tmp_path / "sample.csv", {"nir-a": 5})
    out = tmp_path / "out"
    assert run("--source", src, "--frames-out", out, "--sample", sample,
               "--dry-run").returncode == 0
    assert not out.exists(), "--dry-run 不得寫任何檔案"


def test_existing_complete_dir_is_skipped_not_overwritten(tmp_path):
    """重跑冪等：完整目錄跳過；不完整目錄拒絕覆寫（不靜默毀掉半成品）。"""
    src = tmp_path / "src"
    build_seq_dir(src, "nir-a", n=3)
    out = tmp_path / "out"
    assert run("--source", src, "--frames-out", out).returncode == 0
    assert run("--source", src, "--frames-out", out).returncode == 0  # 冪等

    (out / "nir-a" / "0003.png").unlink()
    r = run("--source", src, "--frames-out", out)
    assert r.returncode == 2 and "拒絕覆寫" in r.stderr


# ── 09-07 官方佈局實看後新增 ─────────────────────────────────────────────────
def build_official_layout(root: Path, *, with_mosaic=True) -> Path:
    """gDrive `ranking/` 的真實形狀：HSI-<模態>[-Falsecolor]/<seq>/0001.jpg + init + groundtruth。"""
    rk = root / "ranking"
    for mod_dir, seqs in (("HSI-NIR-Falsecolor", ["blackball2", "bus1"]),
                          ("HSI-RedNIR-Falsecolor", ["cardpigs1"]),
                          ("HSI-VIS-FalseColor", ["balloon", "ant2"])):
        for s in seqs:
            d = build_seq_dir(rk / mod_dir, s, n=4, names=[f"{i:04d}.jpg" for i in range(1, 5)])
            (d / "groundtruth_rect.txt").write_text("10 10 20 15\n" * 4)
            (d / "description.txt").write_text("Occlusion\n")
    if with_mosaic:
        for mod_dir in ("HSI-NIR", "HSI-RedNIR", "HSI-VIS"):
            d = rk / mod_dir / "whatever"
            d.mkdir(parents=True)
            (d / "0001.png").write_bytes(make_png(8, 8))
            (d / "init_rect.txt").write_text("1 1 2 2\n")
    return rk


def test_official_layout_root_gets_modality_prefix(tmp_path):
    """--source 指到 ranking/ 根：前綴從資料夾名來，mosaic 資料夾略過，GT/description 不進輸出。"""
    rk = build_official_layout(tmp_path)
    out = tmp_path / "out"
    r = run("--source", rk, "--frames-out", out, "--write-sample", out / "sample.csv")
    assert r.returncode == 0, r.stderr + r.stdout
    got = sorted(p.name for p in out.iterdir() if p.is_dir())
    assert got == ["nir-blackball2", "nir-bus1", "rednir-cardpigs1", "vis-ant2", "vis-balloon"]
    assert not (out / "nir-blackball2" / "groundtruth_rect.txt").exists()
    assert not (out / "nir-blackball2" / "description.txt").exists()
    assert (out / "nir-blackball2" / "init_rect.txt").read_text().strip() == "10 10 20 15"
    assert "略過 mosaic" in r.stdout
    lines = (out / "sample.csv").read_text().splitlines()
    assert lines[0] == "ID,x,y,width,height"
    assert lines[1] == "nir-blackball2_1,0,0,0,0" and lines[4] == "nir-blackball2_4,0,0,0,0"
    assert lines[5] == "nir-bus1_1,0,0,0,0"
    assert len(lines) == 1 + 5 * 4
    # 產出的 sample 必須能通過 run_ranking_b 的契約檢查
    sys.path.insert(0, str(SCRIPT.parents[1]))
    from run_ranking_b import _sample_contract  # noqa: E402
    groups, errors = _sample_contract(out / "sample.csv")
    assert errors == [] and sorted(groups) == got


def test_official_layout_single_modality_dir(tmp_path):
    rk = build_official_layout(tmp_path, with_mosaic=False)
    out = tmp_path / "out"
    r = run("--source", rk / "HSI-RedNIR-Falsecolor", "--frames-out", out)
    assert r.returncode == 0, r.stderr
    assert [p.name for p in out.iterdir() if p.is_dir()] == ["rednir-cardpigs1"]


def test_mosaic_dir_is_refused(tmp_path):
    rk = build_official_layout(tmp_path)
    r = run("--source", rk / "HSI-NIR", "--frames-out", tmp_path / "out", "--dry-run")
    assert r.returncode != 0 and "mosaic" in (r.stderr + r.stdout)


def test_16bit_png_is_blocked_even_without_folder_hint(tmp_path):
    src = tmp_path / "src"
    d = src / "nir-x"
    d.mkdir(parents=True)
    png16 = make_png(8, 8)
    png16 = png16[:24] + bytes([16]) + png16[25:]  # 改 IHDR bit depth（CRC 錯無妨，只讀檔頭）
    (d / "0001.png").write_bytes(png16)
    (d / "init_rect.txt").write_text("1 1 2 2\n")
    r = run("--source", src, "--frames-out", tmp_path / "out", "--dry-run")
    assert r.returncode == 2 and "16-bit PNG" in r.stderr


def test_container_dir_is_fatal_not_silently_merged(tmp_path):
    """影像在更深一層、且目錄名不是模態名 ⇒ 以前會 rglob 合併成一支，現在必須 FATAL。"""
    src = tmp_path / "src"
    build_seq_dir(src / "batch1", "a", n=3)
    build_seq_dir(src / "batch1", "b", n=3)
    r = run("--source", src, "--frames-out", tmp_path / "out", "--dry-run")
    assert r.returncode != 0
    assert "容器不是序列" in (r.stderr + r.stdout)


def test_explicit_prefix_and_no_prefix_warning(tmp_path):
    src = tmp_path / "src"
    build_seq_dir(src, "car1", n=3)
    out = tmp_path / "out"
    r = run("--source", src, "--frames-out", out, "--dry-run")
    assert r.returncode == 0 and "沒有模態前綴" in r.stderr
    r = run("--source", src, "--frames-out", out, "--prefix", "nir-")
    assert r.returncode == 0, r.stderr
    assert [p.name for p in out.iterdir() if p.is_dir()] == ["nir-car1"]


def test_write_sample_and_sample_are_exclusive(tmp_path):
    src = tmp_path / "src"
    build_seq_dir(src, "nir-a", n=2)
    smp = write_sample(tmp_path / "s.csv", {"nir-a": 2})
    r = run("--source", src, "--frames-out", tmp_path / "out",
            "--sample", smp, "--write-sample", tmp_path / "w.csv")
    assert r.returncode != 0 and "互斥" in (r.stderr + r.stdout)


def test_drive_folder_zip_keeps_official_layout_prefix(tmp_path):
    """Drive「下載資料夾為 zip」：zip 內是 ranking/HSI-*-Falsecolor/<seq>/…，前綴要從上層來，mosaic 略過。"""
    rk = build_official_layout(tmp_path)
    z = tmp_path / "ranking-20260907T000000Z-1-001.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for p in sorted(rk.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(tmp_path))
    out = tmp_path / "out"
    r = run("--source", z, "--frames-out", out, "--write-sample", out / "s.csv")
    assert r.returncode == 0, r.stderr + r.stdout
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == [
        "nir-blackball2", "nir-bus1", "rednir-cardpigs1", "vis-ant2", "vis-balloon"]
    assert "略過 mosaic" in r.stdout
    assert (out / "nir-blackball2" / "init_rect.txt").read_text().strip() == "10 10 20 15"


def test_single_nested_image_dir_is_accepted(tmp_path):
    """`<seq>/img/0001.jpg` 只多一層、不可能合併 ⇒ 接受，序列名用外層。"""
    src = tmp_path / "src"
    build_seq_dir(src / "nir-a", "img", n=3)
    (src / "nir-a" / "init_rect.txt").write_text("10 10 20 15\n")
    out = tmp_path / "out"
    r = run("--source", src, "--frames-out", out)
    assert r.returncode == 0, r.stderr
    assert [p.name for p in out.iterdir() if p.is_dir()] == ["nir-a"]
    assert len(list((out / "nir-a").glob("*.png"))) == 3
