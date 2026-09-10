from collections import Counter


def validate_alignment(rows: list[dict], layers: int) -> None:
    counts = Counter((row["request_id"], row["iteration"], row["layer"], row["stage"]) for row in rows)
    assert all(value == 1 for value in counts.values())
    groups = {(row["request_id"], row["iteration"]) for row in rows}
    for request, iteration in groups:
        observed = {row["layer"] for row in rows if row["request_id"] == request and row["iteration"] == iteration}
        assert observed == set(range(layers))


def test_trace_alignment_rejects_missing_layer() -> None:
    rows = [
        {"request_id": "r", "iteration": 0, "layer": layer, "stage": "attention"}
        for layer in range(2)
    ]
    validate_alignment(rows, 2)

