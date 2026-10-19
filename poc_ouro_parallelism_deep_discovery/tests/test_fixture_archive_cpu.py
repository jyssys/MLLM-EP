"""Small opaque-byte controls for the lossless fixture archive. No model or GPU test."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

from poc_ouro_parallelism_deep_discovery.fixture_archive import (
    MAX_CHUNK_BYTES, OUTPUTS, ROOT, create_archive, restore_archive, verify_archive)


def synthetic_bytes(length=8192):
    # Deterministic non-model payload. Include every possible byte value.
    state, result = 0x12345678, bytearray(range(256))
    for _ in range(max(0, length - len(result))):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        result.append(state & 255)
    return bytes(result[:length])


class FixtureArchiveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="cpu_fixture_archive_", dir=OUTPUTS)
        self.base = Path(self.directory.name)
        self.fixture = self.base / "rank0" / "fixture.pt"
        self.fixture.parent.mkdir()
        self.payload = synthetic_bytes()
        self.fixture.write_bytes(self.payload)

    def tearDown(self):
        self.directory.cleanup()

    def archive(self, *, chunk_bytes=113):
        manifest = create_archive(self.fixture, chunk_bytes=chunk_bytes)
        directory = self.fixture.parent / "fixture.archive-v1"
        return directory, manifest

    def rewrite_manifest(self, directory, value):
        (directory / "manifest.json").write_text(json.dumps(value))

    def test_multi_chunk_byte_exact_roundtrip_and_original_unchanged(self):
        before = self.fixture.stat()
        directory, manifest = self.archive()
        after = self.fixture.stat()
        self.assertGreater(len(manifest["chunks"]), 2)
        self.assertEqual(manifest["original_bytes"], len(self.payload))
        self.assertEqual(manifest["original_sha256"], hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(manifest["compressed_stream_bytes"], sum(r["bytes"] for r in manifest["chunks"]))
        for row in manifest["chunks"]:
            raw = (directory / row["filename"]).read_bytes()
            self.assertGreater(row["bytes"], 0)
            self.assertLessEqual(row["bytes"], 113)
            self.assertLessEqual(row["bytes"], MAX_CHUNK_BYTES)
            self.assertEqual(len(raw), row["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row["sha256"])
        self.assertEqual(verify_archive(directory / "manifest.json")["status"], "BYTE_EXACT_PASS")
        restored = self.fixture.with_name("restored.pt")
        self.assertEqual(restore_archive(directory / "manifest.json", restored)["status"], "BYTE_EXACT_PASS")
        self.assertEqual(restored.read_bytes(), self.payload)
        self.assertEqual(self.fixture.read_bytes(), self.payload)
        self.assertEqual((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns),
                         (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns))
        self.assertTrue(manifest["original_post_archive_hash_matches"])

    def test_empty_and_one_byte_inputs(self):
        for length in (0, 1):
            with self.subTest(length=length):
                self.fixture.write_bytes(b"x" * length)
                archive = self.fixture.parent / f"empty-control-{length}"
                manifest = create_archive(self.fixture, archive, chunk_bytes=1)
                self.assertEqual(manifest["original_bytes"], length)
                restored = self.fixture.with_name(f"restored-{length}.pt")
                restore_archive(archive / "manifest.json", restored)
                self.assertEqual(restored.read_bytes(), b"x" * length)

    def test_default_limit_is_32mib(self):
        directory, manifest = self.archive(chunk_bytes=MAX_CHUNK_BYTES)
        self.assertEqual(manifest["chunk_bytes_limit"], 33554432)
        self.assertEqual(len(manifest["chunks"]), 1)
        self.assertEqual(verify_archive(directory / "manifest.json")["status"], "BYTE_EXACT_PASS")

    def test_existing_archive_is_never_modified(self):
        directory, manifest = self.archive()
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        with self.assertRaises(FileExistsError):
            create_archive(self.fixture, directory)
        self.assertEqual({p.name: p.read_bytes() for p in directory.iterdir()}, before)
        self.assertEqual(self.fixture.read_bytes(), self.payload)

    def test_existing_restore_target_and_original_are_never_modified(self):
        directory, _ = self.archive()
        for target in (self.fixture, self.fixture.with_name("existing.pt")):
            if target != self.fixture:
                target.write_bytes(b"existing evidence")
            before = target.read_bytes()
            with self.assertRaises(FileExistsError):
                restore_archive(directory / "manifest.json", target)
            self.assertEqual(target.read_bytes(), before)

    def test_corrupt_chunk_does_not_publish_restore(self):
        directory, manifest = self.archive()
        chunk = directory / manifest["chunks"][1]["filename"]
        data = bytearray(chunk.read_bytes())
        data[len(data) // 2] ^= 1
        chunk.write_bytes(data)
        restored = self.fixture.with_name("corrupt-restored.pt")
        with self.assertRaises(ValueError):
            restore_archive(directory / "manifest.json", restored)
        self.assertFalse(restored.exists())
        self.assertFalse(list(restored.parent.glob(".fixture-restore-*.partial")))
        self.assertEqual(self.fixture.read_bytes(), self.payload)

    def test_missing_or_truncated_chunk_is_rejected(self):
        directory, manifest = self.archive()
        chunk = directory / manifest["chunks"][-1]["filename"]
        saved = chunk.read_bytes()
        chunk.write_bytes(saved[:-1])
        with self.assertRaisesRegex(ValueError, "size"):
            verify_archive(directory / "manifest.json")
        chunk.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            verify_archive(directory / "manifest.json")

    def test_manifest_hash_or_original_size_corruption_is_rejected(self):
        directory, original = self.archive()
        for key, value in [("original_sha256", "0" * 64),
                ("compressed_stream_sha256", "0" * 64),
                ("original_bytes", len(self.payload) - 1)]:
            with self.subTest(field=key):
                manifest = dict(original, **{key: value})
                self.rewrite_manifest(directory, manifest)
                with self.assertRaises(ValueError):
                    verify_archive(directory / "manifest.json")

    def test_chunk_order_path_escape_and_excess_limit_are_rejected(self):
        directory, original = self.archive()
        mutations = [lambda m: m["chunks"].reverse(),
            lambda m: m["chunks"][0].update(filename="../fixture.pt"),
            lambda m: m.update(chunk_bytes_limit=MAX_CHUNK_BYTES + 1)]
        for mutate in mutations:
            manifest = json.loads(json.dumps(original))
            mutate(manifest)
            self.rewrite_manifest(directory, manifest)
            with self.assertRaises(ValueError):
                verify_archive(directory / "manifest.json")

    def test_extra_or_truncated_compressed_stream_is_rejected_even_with_updated_chunk_hash(self):
        directory, original = self.archive(chunk_bytes=MAX_CHUNK_BYTES)
        row = original["chunks"][0]
        path = directory / row["filename"]
        raw = path.read_bytes()
        for changed in (raw + b"extra", raw[:-1]):
            with self.subTest(length=len(changed)):
                manifest = json.loads(json.dumps(original))
                path.write_bytes(changed)
                digest = hashlib.sha256(changed).hexdigest()
                manifest["chunks"][0].update(bytes=len(changed), sha256=digest)
                manifest.update(compressed_stream_bytes=len(changed), compressed_stream_sha256=digest)
                self.rewrite_manifest(directory, manifest)
                with self.assertRaises(ValueError):
                    verify_archive(directory / "manifest.json")

    def test_scoped_input_name_and_chunk_limit(self):
        bad = self.fixture.with_name("not-fixture.pt")
        bad.write_bytes(b"not a permitted input name")
        with self.assertRaises(ValueError):
            create_archive(bad)
        with tempfile.TemporaryDirectory(prefix="cpu_archive_bad_scope_", dir=ROOT / "artifacts") as outside:
            raw = Path(outside) / "fixture.pt"
            raw.write_bytes(b"outside outputs")
            with self.assertRaises(ValueError):
                create_archive(raw)
        for count in (0, MAX_CHUNK_BYTES + 1, True):
            with self.assertRaises(ValueError):
                create_archive(self.fixture, chunk_bytes=count)

    def test_symlink_inputs_and_chunks_are_rejected(self):
        linked = self.base / "linked" / "fixture.pt"
        linked.parent.mkdir()
        linked.symlink_to(self.fixture)
        with self.assertRaises(ValueError):
            create_archive(linked)
        directory, manifest = self.archive()
        chunk = directory / manifest["chunks"][0]["filename"]
        replacement = self.fixture.with_name("saved-chunk")
        chunk.rename(replacement)
        chunk.symlink_to(replacement)
        with self.assertRaisesRegex(ValueError, "linked"):
            verify_archive(directory / "manifest.json")

    def test_standalone_cli_and_manifest_reconstruction_command(self):
        script = ROOT / "scripts/c07_fixture_archive.py"
        archive = self.fixture.parent / "archive with spaces"
        result = subprocess.run([sys.executable, str(script), "archive", "--fixture", str(self.fixture),
            "--archive-dir", str(archive), "--chunk-mib", "1"],
            cwd=ROOT / "artifacts", text=True, capture_output=True, check=True)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["status"], "COMPLETE")
        manifest = json.loads((archive / "manifest.json").read_text())
        command = manifest["reconstruction"]["command"]
        self.assertEqual(shlex.split(command), manifest["reconstruction"]["argv"])
        subprocess.run(shlex.split(command), cwd=ROOT / "artifacts", check=True, capture_output=True, text=True)
        self.assertEqual(self.fixture.with_name("fixture.reconstructed.pt").read_bytes(), self.payload)
        check = subprocess.run([sys.executable, str(script), "verify", "--manifest", str(archive / "manifest.json")],
            cwd=ROOT / "artifacts", text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(check.stdout)["status"], "BYTE_EXACT_PASS")
        failed = subprocess.run(shlex.split(command), cwd=ROOT / "artifacts", capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(self.fixture.read_bytes(), self.payload)

    def test_import_does_not_load_torch_or_cuda(self):
        code = ("import sys; import poc_ouro_parallelism_deep_discovery.fixture_archive; "
                "assert 'torch' not in sys.modules; "
                "assert 'libcudart' not in open('/proc/self/maps').read(); "
                "assert 'libcuda.so' not in open('/proc/self/maps').read()")
        subprocess.run([sys.executable, "-c", code], cwd=ROOT.parent,
                       check=True, text=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
