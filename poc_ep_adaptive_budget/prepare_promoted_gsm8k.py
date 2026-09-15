"""Create matched 512-sample and full-test GSM8K promotion sets."""

import hashlib
import json
from pathlib import Path

import pyarrow as pa


ROOT = Path(__file__).resolve().parent
ARROW = Path("/home/esjung/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow")
ANCHOR = ROOT.parent / "poc_llada2_flash_ep" / "data" / "bounded_eval_100"


def main():
    all_rows = pa.ipc.open_stream(str(ARROW)).read_all().to_pylist()
    if len(all_rows) != 1319:
        raise RuntimeError(f"expected official 1319-row test set, got {len(all_rows)}")
    anchor_input = json.loads((ANCHOR / "gsm8k_100.json").read_text())["details"]
    anchor_truth = json.loads((ANCHOR / "gsm8k_100_truth.json").read_text())
    for index in range(100):
        if all_rows[index]["question"] != anchor_input[index]["prompt"]:
            raise RuntimeError(f"prompt mismatch with existing harness at {index}")
        if all_rows[index]["answer"] != anchor_truth[index]["answer"]:
            raise RuntimeError(f"truth mismatch with existing harness at {index}")
    data = ROOT / "data"
    data.mkdir(parents=True, exist_ok=True)
    for size in (512, 1319):
        rows = all_rows[:size]
        prompts = {"details": [{"prompt": row["question"]} for row in rows]}
        truths = [{"id": index, "question": row["question"], "answer": row["answer"]} for index, row in enumerate(rows)]
        (data / f"gsm8k_{size}.json").write_text(json.dumps(prompts, indent=2, ensure_ascii=True) + "\n")
        (data / f"gsm8k_{size}_truth.json").write_text(json.dumps(truths, indent=2, ensure_ascii=True) + "\n")
    manifest = {
        "source": str(ARROW),
        "source_sha256": hashlib.sha256(ARROW.read_bytes()).hexdigest(),
        "sizes": [512, 1319],
        "first_100_prompt_truth_exact_anchor": True,
        "one_sample_pp": {"512": 100 / 512, "1319": 100 / 1319},
    }
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
