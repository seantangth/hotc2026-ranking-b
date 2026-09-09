"""q100 Lambda pilot 的純 CPU/static deployment gates。"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("setup_q100_crop_pilot_lambda_v1.sh")


def test_bash_syntax_and_no_launch_endpoint():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text()
    assert "/instance-operations/launch" not in text
    assert "KEEP_INSTANCE" in text
    assert "RUN_ID" in text
    assert "window_geometry_verdict.json" in text
    assert "D074_cross_environment_confounded" in text
    assert "fresh same-machine q100-444 vs legacy-q95" in text
    assert "1038lab/sam3/resolve/main/sam3.pt" in text
    assert '$MODE/$RUN_ID' in text
    assert "jobs -pr" in text
    assert "sha256sum -c artifact_sha256.txt" in text
    assert '--exclude pilot.log --exclude status.txt' in text
    assert 'rclone copyto "$STATUS"' in text
    assert "selfkill_core.sh 300" in text


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("rankA-historical", "corr=both sam3=legacy-no-explicit-eval-flag samurai=legacy-cross-seq-kf"),
        ("rankB-robust", "corr=top-only sam3=eval samurai=per-sequence-reset"),
    ],
)
def test_validate_only_mode_is_cpu_safe(mode, expected):
    env = os.environ.copy()
    env["PILOT_VALIDATE_ONLY"] = "1"
    result = subprocess.run(
        ["bash", str(SCRIPT), mode], env=env, check=True,
        capture_output=True, text=True,
    )
    assert "VALIDATE_ONLY_PASS" in result.stdout
    assert expected in result.stdout
