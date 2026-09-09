"""Synthetic CPU fixtures test joins only; never research evidence."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_large_matched_transfer as analyzer


class MatchedAnalysisTest(unittest.TestCase):
    def test_request_identity_and_exact_volume(self):
        with tempfile.TemporaryDirectory(prefix="successor_join_test_") as directory:
            root = Path(directory)
            run = root / "b0_clean"
            run.mkdir()
            metadata, records = [], []
            for role, images in (("vision", 4), ("text", 0)):
                rid = f"{role}_00"
                metadata.append({"request_id": rid, "matched_pair": "short_00",
                                 "processor_prompt_tokens": 1800,
                                 "vision_tokens": 1700 if images else 0})
                records.append({"request_id": f"b0_clean:r0:{role}:b1:dp0:{rid}",
                                "warmup": False, "label": f"r0:{role}:b1",
                                "dp_rank": 0, "image_count": images, "family": role,
                                "prompt_tokens": 1800, "e2e_s": 4.0, "ttft_s": .5,
                                "processor_s": .1, "engine_e2e_s": 3.9, "tpot_s": .1})
            inputs = root / "inputs.json"
            inputs.write_text(json.dumps({"requests": metadata}))
            (run/"requests_dp0.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
            with patch("sys.argv", ["analysis", str(root), "--inputs", str(inputs)]), contextlib.redirect_stdout(io.StringIO()):
                analyzer.main()
            summary = json.loads((root/"matched_composition_summary.json").read_text())
            self.assertEqual(summary["matched_pairs"], 1)
            self.assertEqual(summary["exact_token_pairs"], 1)
            self.assertEqual(summary["processor_prediction_mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
