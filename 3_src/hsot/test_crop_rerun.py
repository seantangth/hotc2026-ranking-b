"""crop_rerun JPEG profile 回歸測試（合成影像、純 CPU）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, JpegImagePlugin

from hsot import crop_rerun


def _run_prep(tmp_path: Path, monkeypatch, profile: str | None) -> tuple[list[dict], dict, Path]:
    """用一幀小目標序列跑 prep，同時攔截實際傳給 Pillow 的 kwargs。"""
    frames = tmp_path / "frames" / "tiny"
    frames.mkdir(parents=True)
    src = Image.new("RGB", (256, 192), (20, 80, 160))
    src.save(frames / "000001.jpg", quality=90)

    base_csv = tmp_path / "base.csv"
    pd.DataFrame([
        {"ID": "tiny_1", "x": 100.0, "y": 70.0, "width": 8.0, "height": 8.0},
    ]).to_csv(base_csv, index=False)
    out_root = tmp_path / "crop"
    meta_path = tmp_path / "meta.json"

    calls: list[dict] = []
    original_save = Image.Image.save

    def recording_save(self, fp, *args, **kwargs):
        calls.append(dict(kwargs))
        return original_save(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", recording_save)
    argv = [
        "crop_rerun", "prep",
        "--frames-root", str(tmp_path / "frames"),
        "--base-csv", str(base_csv),
        "--out-root", str(out_root),
        "--meta", str(meta_path),
    ]
    if profile is not None:
        argv += ["--jpeg-profile", profile]
    monkeypatch.setattr(sys, "argv", argv)
    crop_rerun.main()

    return calls, json.loads(meta_path.read_text()), out_root / "tiny" / "000001.jpg"


def test_prep_default_preserves_legacy_q95(monkeypatch, tmp_path):
    calls, meta, output = _run_prep(tmp_path, monkeypatch, profile=None)

    # 歷史路徑不可偷偷多傳 subsampling，否則不再是原本的編碼呼叫。
    assert calls == [{"quality": 95}]
    assert output.is_file()
    assert meta["tiny"]["crop_image_encoding"] == {
        "profile": "legacy-q95",
        "format": "JPEG",
        "quality": 95,
        "subsampling": "pillow-default",
    }


def test_prep_q100_444_passes_explicit_pillow_subsampling(monkeypatch, tmp_path):
    calls, meta, output = _run_prep(tmp_path, monkeypatch, profile="q100-444")

    assert calls == [{"quality": 100, "subsampling": 0}]
    assert meta["tiny"]["crop_image_encoding"] == {
        "profile": "q100-444",
        "format": "JPEG",
        "quality": 100,
        "subsampling": 0,
        "chroma_subsampling": "4:4:4",
    }
    # 不只驗 kwargs：也驗證實際 JPEG 的 sampling factor 是 4:4:4（0）。
    with Image.open(output) as encoded:
        assert JpegImagePlugin.get_sampling(encoded) == 0
