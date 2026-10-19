"""CPU-only tests for saved graph provenance. No Torch import."""

import gzip
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = (Path(__file__).resolve().parents[1] / "scripts" /
          "validate_storage_graph_provenance_cpu_v1.py")
SPEC = importlib.util.spec_from_file_location("storage_graph_provenance", SCRIPT)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def graph(size, mode, parallel=2):
    layers = {"1.4B": 24, "2.6B": 48}[size]
    depths = [0] if mode == "alias" else range(4)
    arguments = [
        ("L_self_modules_embed_tokens_parameters_weight_", f"bf16[{49152 // parallel},2048]"),
        ("L_self_modules_norm_parameters_weight_", "bf16[2048]"),
    ]
    for depth in depths:
        for layer in range(layers):
            for role, shape in sorted(CHECK.expected_shapes(parallel).items()):
                arguments.append((f"L_self_modules_campaign_execution_modules_{depth}_"
                                  f"modules_{layer}_modules_{role}_parameters_weight_", shape))
    signature = ", ".join(f'{name}: "{shape}"' for name, shape in arguments)
    # Every argument appears in the body. The fixture does not execute.
    body = ", ".join(name for name, shape in arguments)
    return f"class GraphModule:\n    def forward(self, {signature}):\n        return ({body})\n".encode()


class ProvenanceTests(unittest.TestCase):
    def test_all_model_arm_counts(self):
        for size, alias, copies in [("1.4B", 194, 770), ("2.6B", 386, 1538)]:
            for mode, count in [("alias", alias), ("copies", copies)]:
                with self.subTest(size=size, mode=mode):
                    result = CHECK.inspect_graph(graph(size, mode), size, mode, 2)
                    self.assertEqual(result["status"], "PASS")
                    self.assertEqual(result["top_weight_arguments"], count)

    def test_wrong_arm_fails(self):
        result = CHECK.inspect_graph(graph("1.4B", "alias"), "1.4B", "copies", 2)
        self.assertIn("STORAGE_DEPTH_PREFIX_COUNTS", result["failures"])

    def test_matching_count_wrong_depth_fails(self):
        payload = graph("1.4B", "copies").replace(b"execution_modules_3_", b"execution_modules_4_")
        result = CHECK.inspect_graph(payload, "1.4B", "copies", 2)
        self.assertEqual(result["top_weight_arguments"], 770)
        self.assertIn("STORAGE_DEPTH_PREFIX_COUNTS", result["failures"])

    def test_unknown_names_fail(self):
        payload = graph("1.4B", "alias").replace(b"campaign_execution", b"unknown_execution")
        result = CHECK.inspect_graph(payload, "1.4B", "alias", 2)
        self.assertIn("UNRECOGNIZED_WEIGHT_NAME", result["failures"])

    def test_export_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = graph("1.4B", "alias")
            for rank in range(2):
                source = root / "cache" / f"rank_{rank}_0" / "backbone" / "computation_graph.py"
                source.parent.mkdir(parents=True)
                source.write_bytes(payload)
            manifest = CHECK.export_job(root / "cache", "1.4B", "alias", 2, root / "export")
            self.assertEqual(manifest["status"], "PASS")
            for record in manifest["records"]:
                compressed = Path(record["gzip_path"]).read_bytes()
                self.assertEqual(gzip.decompress(compressed), payload)
                self.assertEqual(CHECK.sha256(compressed), record["gzip_sha256"])
                self.assertEqual(CHECK.sha256(payload), record["source_sha256"])
            with self.assertRaises(FileExistsError):
                CHECK.export_job(root / "cache", "1.4B", "alias", 2, root / "export")

    def test_rank_mismatch_preserves_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cache").mkdir()
            manifest = CHECK.export_job(root / "cache", "1.4B", "alias", 2, root / "export")
            self.assertEqual(manifest["status"], "FAIL")
            self.assertTrue((root / "export" / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
