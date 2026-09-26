from __future__ import annotations

from pathlib import Path

import pytest

from poc_mllm_ep_foundation.evaluation.workloads import build_sanity_manifest, load_manifest


def test_sanity_manifest_images_exist_and_missing_control_is_loud(tmp_path: Path) -> None:
    path = tmp_path / "sanity.json"
    manifest = build_sanity_manifest(path)
    assert len(manifest["requests"]) <= 16
    assert all(Path(row["image_path"]).exists() for row in manifest["requests"])
    assert not Path(manifest["negative_controls"]["missing_image"]["path"]).exists()
    assert len(load_manifest(path)["requests"]) == len(manifest["requests"])

