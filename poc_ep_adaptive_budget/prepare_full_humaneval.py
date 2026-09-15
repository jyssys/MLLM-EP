"""Build the full 164-task HumanEval set from OpenAI's official artifact."""

import gzip
import hashlib
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
ANCHOR = ROOT.parent / "poc_llada2_flash_ep" / "data" / "bounded_eval_100"


def main():
    data = ROOT / "data"
    data.mkdir(parents=True, exist_ok=True)
    archive = data / "HumanEval.jsonl.gz"
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=30) as response:
            body = response.read()
        archive.write_bytes(body)
    body = archive.read_bytes()
    problems = [json.loads(line) for line in gzip.decompress(body).decode().splitlines() if line]
    if len(problems) != 164:
        raise RuntimeError(f"official HumanEval expected 164 tasks, found {len(problems)}")
    input_anchor = json.loads((ANCHOR / "humaneval_100.json").read_text())["details"]
    truth_anchor = json.loads((ANCHOR / "humaneval_100_truth.json").read_text())
    for index in range(100):
        if problems[index]["prompt"] != input_anchor[index]["prompt"]:
            raise RuntimeError(f"prompt mismatch with existing harness at {index}")
        if problems[index]["test"] != truth_anchor[index]["test"]:
            raise RuntimeError(f"test mismatch with existing harness at {index}")
    inputs = {"details": [{"prompt": problem["prompt"]} for problem in problems]}
    truths = [{"id": index, "prompt": problem["prompt"], "test": problem["test"],
               "entry_point": problem["entry_point"]} for index, problem in enumerate(problems)]
    (data / "humaneval_164.json").write_text(json.dumps(inputs, indent=2, ensure_ascii=True) + "\n")
    (data / "humaneval_164_truth.json").write_text(json.dumps(truths, indent=2, ensure_ascii=True) + "\n")
    manifest = {"source": URL, "source_sha256": hashlib.sha256(body).hexdigest(),
                "n": 164, "first_100_prompt_test_exact_anchor": True, "one_sample_pp": 100 / 164}
    (data / "humaneval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
