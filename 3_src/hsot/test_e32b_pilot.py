from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from e32b_common import (  # noqa: E402
    EXECUTION_MODE,
    EXPERIMENT,
    RESULT_SCHEMA_VERSION,
    exact_frozen_runs,
    frame_tree_identity,
    load_submission,
    manifest_records,
    payload_sha256,
    validate_manifest_structure,
)


MANIFEST = ROOT / "3_src/peft/e32b_frames.json"
A_V023 = ROOT / "5_outputs/submissions/sub_v023_cropwiden55.csv"
A_E46 = ROOT / "5_outputs/e46_sam3_crop_scores_20260819/test_merged.csv"
B_V012 = ROOT / "5_outputs/submissions/sub_v012_e23b_sam21_ablation.csv"
BASE_V067 = ROOT / "5_outputs/submissions/sub_v067_v056_firstframe_init.csv"
SAMPLE = ROOT / "1_data/raw/sample_submisson.csv"
VERDICT = HERE / "e32b_verdict.py"
PROBE = ROOT / "3_src/peft/e32b_exemplar_probe_cuda.py"
LAUNCH = ROOT / "3_src/launch/setup_e32b_k24_pilot_v1.sh"

# E32b 是已結案的歷史探針（08-24），它的判決測試需要 5_outputs/ 的歷史 CSV；
# 交付包（09-06 起）刻意不含 5_outputs ⇒ 在包內整檔 skip，不是失敗（D093 已記：要 ignore）。
pytestmark = pytest.mark.skipif(
    not (A_V023.is_file() and B_V012.is_file() and BASE_V067.is_file()),
    reason="historical 5_outputs CSVs are not shipped in the delivery package")


def load_manifest() -> dict:
    value = json.loads(MANIFEST.read_text())
    validate_manifest_structure(value)
    return value


def pool_count(a_path: Path) -> tuple[int, int]:
    a, _ = load_submission(a_path)
    b, _ = load_submission(B_V012)
    n = 0
    seqs = 0
    for seq in a:
        ra = exact_frozen_runs(a[seq])
        rb = exact_frozen_runs(b[seq])
        frames = [f for f in a[seq] if ra[f] >= 24 and rb[f] >= 2]
        n += len(frames)
        seqs += bool(frames)
    return n, seqs


def synthetic_probe(manifest: dict, *, healthy_detect: bool = True) -> dict:
    target_ids = {r["id"] for r in manifest_records(manifest, "target")}
    records = []
    for source in manifest_records(manifest):
        cohort = "target" if source["id"] in target_ids else "healthy_proxy"
        detected = cohort == "target" or healthy_detect
        if cohort == "target":
            det = [source["a_box"][0] + 2.0, *source["a_box"][1:]]
        else:
            det = list(source["a_box"])
        records.append({
            "id": source["id"],
            "cohort": cohort,
            "a_box": source["a_box"],
            "b_box": source["b_box"],
            "n_det": 1 if detected else 0,
            "det_box": det if detected else None,
            "det_score": 0.99 if detected else None,
        })
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "execution_mode": EXECUTION_MODE,
        "true_tracker_reinit": False,
        "manifest_payload_sha256": manifest["payload_sha256"],
        "runtime_fallbacks": 0,
        "n_errors": 0,
        "records": records,
    }
    result["payload_sha256"] = payload_sha256(result)
    return result


def test_canonical_and_fresh_a_are_distinct_pools():
    assert pool_count(A_V023) == (125, 5)
    assert pool_count(A_E46) == (120, 5)


def test_manifest_locks_exact_counts_and_controls():
    manifest = load_manifest()
    assert manifest["counts"] == {
        "target": 125,
        "healthy_proxy": 40,
        "target_sequences": 5,
    }
    assert manifest["policy"]["true_tracker_reinit"] is False
    assert manifest["policy"]["execution_mode"] == EXECUTION_MODE
    assert all(len(manifest["sequences"][seq]["healthy_proxy"]) == 8
               for seq in manifest["sequence_order"])


def test_ready_verdict_changes_only_125_target_rows(tmp_path: Path):
    manifest = load_manifest()
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps(synthetic_probe(manifest)))
    candidate = tmp_path / "candidate.csv"
    report = tmp_path / "verdict.json"
    completed = subprocess.run([
        sys.executable, str(VERDICT),
        "--manifest", str(MANIFEST),
        "--probe-result", str(probe),
        "--base", str(BASE_V067),
        "--sample", str(SAMPLE),
        "--candidate-out", str(candidate),
        "--report", str(report),
    ], text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    verdict = json.loads(report.read_text())
    assert verdict["verdict"] == "READY_FOR_ONE_LB_OFFLINE_AB"
    assert verdict["auc_gain_measured"] is False
    assert verdict["true_tracker_reinit"] is False
    assert verdict["candidate"]["detector_replacements"] == 125

    base, base_order = load_submission(BASE_V067)
    out, out_order = load_submission(candidate)
    assert out_order == base_order
    changed = {
        f"{seq}_{frame}"
        for seq in base for frame in base[seq]
        if base[seq][frame] != out[seq][frame]
    }
    assert changed == {r["id"] for r in manifest_records(manifest, "target")}
    assert all(out[seq][min(out[seq])] == base[seq][min(base[seq])] for seq in base)


def test_no_go_proxy_removes_stale_candidate(tmp_path: Path):
    manifest = load_manifest()
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps(synthetic_probe(manifest, healthy_detect=False)))
    candidate = tmp_path / "candidate.csv"
    candidate.write_text("stale\n")
    report = tmp_path / "verdict.json"
    completed = subprocess.run([
        sys.executable, str(VERDICT),
        "--manifest", str(MANIFEST),
        "--probe-result", str(probe),
        "--base", str(BASE_V067),
        "--sample", str(SAMPLE),
        "--candidate-out", str(candidate),
        "--report", str(report),
    ], text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    verdict = json.loads(report.read_text())
    assert verdict["verdict"] == "NO_GO_PROXY_SANITY"
    assert verdict["candidate"] is None
    assert not candidate.exists()


def test_assemble_requires_all_exact_sequence_artifacts(tmp_path: Path):
    manifest = load_manifest()
    synthetic = synthetic_probe(manifest)
    by_id = {r["id"]: r for r in synthetic["records"]}
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    revision = "96914d2425f90a64f45ca977c2b5165418099543"
    for seq in manifest["sequence_order"]:
        expected = manifest["sequences"][seq]["target"] + manifest["sequences"][seq]["healthy_proxy"]
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "execution_mode": EXECUTION_MODE,
            "true_tracker_reinit": False,
            "sequence": seq,
            "manifest_payload_sha256": manifest["payload_sha256"],
            "sam3_revision": revision,
            "checkpoint_sha256": "a" * 64,
            "runtime_fallbacks": 0,
            "n_errors": 0,
            "elapsed_seconds": 1.0,
            "records": [by_id[r["id"]] for r in expected],
        }
        result["payload_sha256"] = payload_sha256(result)
        (records_dir / f"{seq}.json").write_text(json.dumps(result))
    completed = subprocess.run([
        sys.executable, str(PROBE),
        "--manifest", str(MANIFEST),
        "--out-dir", str(tmp_path),
        "--assemble-only",
    ], text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assembled = json.loads((tmp_path / "probe_result.json").read_text())
    assert len(assembled["records"]) == 165
    assert assembled["runtime_fallbacks"] == 0
    assert assembled["true_tracker_reinit"] is False


def test_frame_content_identity_catches_same_size_mutation(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "0001.jpg").write_bytes(b"abcd")
    (second / "0001.jpg").write_bytes(b"wxyz")
    a = frame_tree_identity(first)
    b = frame_tree_identity(second)
    assert a["manifest_sha256"] == b["manifest_sha256"]
    assert a["content_manifest_sha256"] != b["content_manifest_sha256"]


def test_shared_reuse_launch_has_cpu_safe_static_gate():
    subprocess.run(["bash", "-n", str(LAUNCH)], check=True)
    text = LAUNCH.read_text()
    assert "/instance-operations/launch" not in text
    for token in (
        "E32B_REUSE_SHARED",
        "E32B_SHARED_PY3",
        "E32B_SHARED_SAM3_CHECKPOINT",
        "E32B_SHARED_TEST_FRAMES_ROOT",
        "--reuse-extracted-frames",
        "--expected-checkpoint-sha256",
        "--expected-sam3-revision",
        "shared reuse requires explicit KEEP_INSTANCE=1",
    ):
        assert token in text

    env = os.environ.copy()
    env.update({"INSTANCE_ID": "static-validation", "E32B_VALIDATE_ONLY": "1"})
    standalone = subprocess.run(
        ["bash", str(LAUNCH)], env=env, text=True, capture_output=True, check=False
    )
    assert standalone.returncode == 0, standalone.stderr + standalone.stdout
    assert "E32B_VALIDATE_ONLY_PASS reuse=0 keep=0" in standalone.stdout

    env.update({"E32B_REUSE_SHARED": "1", "KEEP_INSTANCE": "1"})
    shared = subprocess.run(
        ["bash", str(LAUNCH)], env=env, text=True, capture_output=True, check=False
    )
    assert shared.returncode == 0, shared.stderr + shared.stdout
    assert "E32B_VALIDATE_ONLY_PASS reuse=1 keep=1" in shared.stdout

    env["KEEP_INSTANCE"] = "0"
    unsafe = subprocess.run(
        ["bash", str(LAUNCH)], env=env, text=True, capture_output=True, check=False
    )
    assert unsafe.returncode == 2
    assert "shared reuse requires KEEP_INSTANCE=1" in unsafe.stderr
