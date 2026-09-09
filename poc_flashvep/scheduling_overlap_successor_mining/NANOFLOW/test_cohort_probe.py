"""Parent-loop identity regression; native numerical correctness stays separate."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from cohort_probe import run


class Queue:
    def __init__(self):
        self.history = []

    def put(self, value):
        self.inputs = value[0]
        self.history.append(value[0])

    def get(self, timeout):
        return [(index, [41]) for index, _ in self.inputs]


queue = Queue()
entry = SimpleNamespace(command=SimpleNamespace(value=b""), decode_bts=SimpleNamespace(value=0),
                        request_queues=[queue], result_queue=queue, barrier=None,
                        step_barrier=lambda barrier: None, processes=[],
                        terminate_workers=lambda processes, barrier: None,
                        arts=SimpleNamespace(tokenizer=SimpleNamespace(
                            tokenizer=SimpleNamespace(decode=lambda ids: "recorded"))))
with tempfile.TemporaryDirectory(prefix="nano-cohort-test-") as tmp:
    path, output = Path(tmp) / "trace.jsonl", Path(tmp) / "result.json"
    path.write_text("".join(json.dumps({"request_id": str(i), "input_ids": [1] * 32,
                                         "max_new_tokens": 4}) + "\n" for i in range(2)))
    run(entry, path, output)
    result = json.loads(output.read_text())
    assert len(result["requests"]) == 2
    assert {r["request_id"] for r in result["requests"]} == {"0", "1"}
    for row in result["requests"]:
        assert row["output_ids"] == [41] * 4
        assert len(row["token_times_s"]) == 4
        assert 0 <= row["ttft_s"] <= row["e2e_s"]
        assert row["itl_s"] and all(v >= 0 for v in row["itl_s"])
    for row in result["requests"]:
        row["output_ids"] = [90, 91, 92, 93]
    forced = Path(tmp) / "forced.json"
    forced.write_text(json.dumps(result))
    queue.history.clear()
    run(entry, path, Path(tmp) / "forced_result.json", forced)
    checked = json.loads((Path(tmp) / "forced_result.json").read_text())
    assert not checked["performance_eligible"]
    assert queue.history[-3:] == [[(4, [90]), (5, [90])],
                                 [(4, [91]), (5, [91])],
                                 [(4, [92]), (5, [92])]]
    assert all(r["output_ids"] == [41] * 4 for r in checked["requests"])
    path.write_text("".join(json.dumps({"request_id": str(i), "input_ids": [1] * 32,
                                         "max_new_tokens": 1}) + "\n" for i in range(2)))
    queue.history.clear()
    prefill = Path(tmp)/"prefill.json"
    run(entry, path, prefill)
    checked_prefill = json.loads(prefill.read_text())
    assert all(r["e2e_s"] == r["ttft_s"] and r["tpot_s"] is None and r["itl_s"] == []
               for r in checked_prefill["requests"])
print("COHORT_PARENT_IDENTITY_PASS; no CUDA imported")
