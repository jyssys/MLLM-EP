"""CPU-only checks for source-row and expert-owner reconstruction."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_atlas.py"
spec = importlib.util.spec_from_file_location("analyze_atlas", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_source_row_and_owner_mapping():
    record = {
        "request_id": "measured_0",
        "block_starts": [0, 0, 0, 0],
        "batch_row": 3,
        "block_length": 32,
        "ep_latest_invocation": 5,
    }
    ids = [0, 1, 64, 65, 128, 129, 192, 193]
    routes = {
        ("measured_0", 3, 16, 5): {
            "topk_ids": [ids for _ in range(32)]
        }
    }
    loads, observed_ids, source = module.token_signature(record, routes, (16,), 7)
    assert source == 3
    assert observed_ids == ids
    assert loads == [2, 2, 2, 2]
    assert sum(loads) == 8


def test_missing_route_cannot_be_silently_imputed():
    record = {
        "request_id": "measured_0",
        "block_starts": [0, 0, 0, 0],
        "batch_row": 0,
        "block_length": 32,
        "ep_latest_invocation": 5,
    }
    assert module.token_signature(record, {}, (16,), 0) is None
