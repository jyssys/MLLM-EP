# Lossless C07 fixture archive

## Result

The separate helper keeps the original ignored `fixture.pt` unchanged.
It compresses the complete byte stream with zlib.
It splits that stream into chunks of at most 32 MiB, or 33,554,432 bytes.
It never loads tensors or changes numerical values.

The [helper](../fixture_archive.py), [standalone CLI](../scripts/c07_fixture_archive.py), and [CPU tests](../tests/test_fixture_archive_cpu.py) are new files.
This work changes no C07 worker, runner, installed package, or prior evidence file.
No GPU execution or actual model-fixture archive ran during this task.

## Archive contract

Creation accepts an explicit regular file named `fixture.pt` inside this campaign's `outputs` directory.
The caller must name one file.
The helper does not discover or scan probe outputs.
It rejects symbolic-link inputs and paths outside that directory.

The helper creates a new archive directory with exclusive creation.
An existing directory causes failure, including an incomplete archive directory.
An interrupted archive can leave partial chunks without a complete manifest.
Keep those files for diagnosis or choose a new archive name.
The helper never removes or overwrites an earlier archive.

The manifest records:

- The original path, byte count, and SHA256 hash.
- The codec, compression level, zlib versions, and chunk limit.
- The compressed-stream byte count and SHA256 hash.
- Each chunk's ordered filename, byte count, and SHA256 hash.
- An exact reconstruction command and its argument list.
- The successful readback check and the final original-file hash check.

Creation reads back and decodes all chunks before it writes the complete manifest.
It then checks the original content hash and file identity again.
These checks reject a detected source change during archiving.
They do not provide a filesystem snapshot against arbitrary concurrent writers.
Run the helper only after fixture export finishes.

## Commands

These commands are templates, not executed model-fixture results.
Replace the explicit path with one completed rank fixture.

```bash
python -m poc_ouro_parallelism_deep_discovery.scripts.c07_fixture_archive archive --fixture /absolute/campaign/outputs/job/model/rank0/fixture.pt
```

The default archive directory is `fixture.archive-v1` beside that file.
Use `--archive-dir` to select a different new directory inside campaign `outputs`.
The optional `--chunk-mib` value must be an integer from 1 through 32.

The CLI also supports `verify --manifest /absolute/path/manifest.json`.
Verification checks every chunk and the complete decoded byte stream.
It creates no restored file.

Use the manifest's `reconstruction.command` for exact reconstruction in this workspace.
That command uses absolute paths and runs from any working directory.
Restoration checks the hashes before it publishes the output path.
An existing output path causes failure.
The original `fixture.pt` therefore cannot be overwritten through restoration.
The reconstructed bytes must match the recorded original byte count and SHA256 hash.

## CPU evidence and limits

The CPU suite contains 14 tests.
The fixtures contain deterministic opaque bytes, not model weights or model outputs.
The suite checks multi-chunk roundtrip, an exact CLI reconstruction command, and empty and one-byte inputs.
It also checks overwrite refusal, corrupt or missing chunks, truncated streams, trailing data, path scope, and symbolic links.
Failed restoration publishes no output file.
Import checks find neither Torch nor CUDA in the test process.

The [validation ledger](../artifacts/c07_fixture_archive_cpu_validation_v1.json) records actual commands, results, source hashes, and work timestamps.
The helper uses bounded stream reads and bounded decode output.
Its archive work includes compression, complete readback, decompression, hashing, and filesystem synchronization.
Those costs remain outside the fixture's physical-service interval.
This task measures no model-fixture compression ratio, storage saving, model quality, or GPU speedup.

Retain the original ignored file as raw evidence.
Use the chunks and manifest when evidence must fit repository blob limits.
Reconstruction preserves the original container bytes, including all numerical and metadata evidence.
