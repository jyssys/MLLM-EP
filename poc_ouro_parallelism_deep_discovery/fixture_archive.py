"""Lossless byte archives for explicit campaign fixture.pt files. No Torch import."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import sys
import tempfile
import zlib


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
MAX_CHUNK_BYTES = 32 * 1024**2
IO_BYTES = 1024**2
FORMAT = "ouro-fixture-zlib-chunks-v1"


def scoped_path(path, *, existing=False):
    """Resolve an explicit path within this campaign's outputs directory."""
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ValueError("A fixture archive path must not be a symbolic link.")
    resolved = path.resolve(strict=existing)
    if not resolved.is_relative_to(OUTPUTS) or resolved == OUTPUTS:
        raise ValueError("Use an explicit path within this campaign's outputs directory.")
    return resolved


def file_hash(path):
    result, length = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while block := stream.read(IO_BYTES):
            result.update(block)
            length += len(block)
    return length, result.hexdigest()


def identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class ChunkWriter:
    def __init__(self, directory, limit):
        self.directory, self.limit = directory, limit
        self.stream, self.rows = None, []
        self.total, self.stream_hash = 0, hashlib.sha256()

    def write(self, data):
        view = memoryview(data)
        while view:
            if self.stream is None:
                name = f"chunk-{len(self.rows):06d}.zlibpart"
                self.stream = (self.directory / name).open("xb")
                self.current = dict(index=len(self.rows), filename=name, bytes=0)
                self.chunk_hash = hashlib.sha256()
            length = min(len(view), self.limit - self.current["bytes"])
            part, view = view[:length], view[length:]
            if self.stream.write(part) != length:
                raise OSError("A fixture archive chunk write was incomplete.")
            self.chunk_hash.update(part)
            self.stream_hash.update(part)
            self.current["bytes"] += length
            self.total += length
            if self.current["bytes"] == self.limit:
                self.close_chunk()

    def close_chunk(self):
        if self.stream is not None:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
            self.stream = None
            self.current["sha256"] = self.chunk_hash.hexdigest()
            self.rows.append(self.current)

    def close_partial(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def validate_manifest(manifest):
    """Check the fixed format and ordered local chunk names before reading chunks."""
    if manifest.get("format") != FORMAT or manifest.get("schema_version") != 1:
        raise ValueError("The fixture archive format is not supported.")
    if manifest.get("status") != "COMPLETE" or manifest.get("compression") != {
            "codec": "zlib", "wbits": zlib.MAX_WBITS, "level": 6}:
        raise ValueError("The fixture archive is incomplete or has an unsupported codec.")
    limit, rows = manifest.get("chunk_bytes_limit"), manifest.get("chunks")
    if type(limit) is not int or not 1 <= limit <= MAX_CHUNK_BYTES:
        raise ValueError("The fixture archive chunk limit exceeds 32 MiB or is invalid.")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100000:
        raise ValueError("The fixture archive requires a bounded nonempty chunk list.")
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or type(row.get("index")) is not int or row["index"] != index
                or row.get("filename") != f"chunk-{index:06d}.zlibpart"
                or type(row.get("bytes")) is not int or not 1 <= row["bytes"] <= limit):
            raise ValueError("A fixture archive chunk name, order, or size is invalid.")
        if index < len(rows) - 1 and row["bytes"] != limit:
            raise ValueError("A non-final fixture archive chunk is not full.")
    for field in ("original_bytes", "compressed_stream_bytes"):
        if type(manifest.get(field)) is not int or manifest[field] < 0:
            raise ValueError("The fixture archive byte counts are invalid.")
    if sum(row["bytes"] for row in rows) != manifest["compressed_stream_bytes"]:
        raise ValueError("The compressed byte count differs from the chunk list.")
    for digest in [manifest.get("original_sha256"), manifest.get("compressed_stream_sha256")] + [r.get("sha256") for r in rows]:
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("A fixture archive SHA256 value is invalid.")


def read_manifest(path):
    path = scoped_path(path, existing=True)
    if path.name != "manifest.json" or not path.is_file() or path.stat().st_size > 16 * 1024**2:
        raise ValueError("Use an explicit manifest.json of at most 16 MiB.")
    with path.open("rb") as stream:
        raw = stream.read(16 * 1024**2 + 1)
    if len(raw) > 16 * 1024**2:
        raise ValueError("The fixture archive manifest grew beyond 16 MiB.")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("The fixture archive manifest must be an object.")
    validate_manifest(manifest)
    return path, manifest


def check_stream(directory, manifest, output=None):
    """Check chunk hashes and decode bounded blocks without loading tensor objects."""
    validate_manifest(manifest)
    decoder = zlib.decompressobj(wbits=zlib.MAX_WBITS)
    original_hash, compressed_hash = hashlib.sha256(), hashlib.sha256()
    original_bytes, compressed_bytes = 0, 0

    def consume(block):
        nonlocal original_bytes
        original_bytes += len(block)
        if original_bytes > manifest["original_bytes"]:
            raise ValueError("The decoded fixture exceeds its declared byte count.")
        original_hash.update(block)
        if output is not None and output.write(block) != len(block):
            raise OSError("A restored fixture write was incomplete.")

    for row in manifest["chunks"]:
        path = directory / row["filename"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError("A fixture archive chunk is missing, linked, or has an incorrect size.")
        digest, count = hashlib.sha256(), 0
        with path.open("rb") as stream:
            while block := stream.read(IO_BYTES):
                count += len(block)
                if count > row["bytes"]:
                    raise ValueError("A fixture archive chunk changed during reading.")
                digest.update(block)
                compressed_hash.update(block)
                compressed_bytes += len(block)
                pending = block
                while pending:
                    try:
                        decoded = decoder.decompress(pending, IO_BYTES)
                    except zlib.error as error:
                        raise ValueError("The fixture archive compressed stream is corrupt.") from error
                    consume(decoded)
                    pending = decoder.unconsumed_tail
                    if decoder.unused_data:
                        raise ValueError("The fixture archive contains trailing or concatenated compressed data.")
        if count != row["bytes"] or digest.hexdigest() != row["sha256"]:
            raise ValueError("A fixture archive chunk SHA256 or byte count differs.")
    # Drain any bounded output retained after a max_length decode.
    while True:
        decoded = decoder.decompress(b"", IO_BYTES)
        if not decoded:
            break
        consume(decoded)
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("The fixture archive compressed stream is incomplete.")
    if (compressed_bytes != manifest["compressed_stream_bytes"]
            or compressed_hash.hexdigest() != manifest["compressed_stream_sha256"]
            or original_bytes != manifest["original_bytes"]
            or original_hash.hexdigest() != manifest["original_sha256"]):
        raise ValueError("The fixture archive stream byte count or SHA256 differs.")
    return dict(status="BYTE_EXACT_PASS", original_bytes=original_bytes,
                original_sha256=original_hash.hexdigest(), compressed_stream_bytes=compressed_bytes,
                compressed_stream_sha256=compressed_hash.hexdigest(), chunks=len(manifest["chunks"]))


def create_archive(fixture, archive_directory=None, *, chunk_bytes=MAX_CHUNK_BYTES):
    fixture = scoped_path(fixture, existing=True)
    if fixture.name != "fixture.pt" or not fixture.is_file():
        raise ValueError("The input must be an explicit regular fixture.pt file.")
    if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= MAX_CHUNK_BYTES:
        raise ValueError("The chunk limit must be within 1 byte..32 MiB.")
    directory = scoped_path(archive_directory or fixture.parent / "fixture.archive-v1")
    if not directory.parent.is_dir():
        raise ValueError("The explicit archive parent directory must already exist.")
    # Atomic exclusive creation. Never merge, overwrite, or remove a previous archive.
    directory.mkdir(exist_ok=False)
    writer = ChunkWriter(directory, chunk_bytes)
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(fixture, flags), "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("The fixture input is not a regular file.")
            compressor, digest, length = zlib.compressobj(level=6, wbits=zlib.MAX_WBITS), hashlib.sha256(), 0
            while block := stream.read(IO_BYTES):
                digest.update(block)
                length += len(block)
                writer.write(compressor.compress(block))
            writer.write(compressor.flush(zlib.Z_FINISH))
            writer.close_chunk()
            if identity(before) != identity(os.fstat(stream.fileno())) or length != before.st_size:
                raise ValueError("The fixture changed during compression. Keep the partial archive for diagnosis.")
        command = [sys.executable, str(ROOT / "scripts/c07_fixture_archive.py"), "restore", "--manifest",
                   str(directory / "manifest.json"), "--output", str(fixture.with_name("fixture.reconstructed.pt"))]
        manifest = dict(schema_version=1, format=FORMAT, status="COMPLETE",
            created_UTC=datetime.now(timezone.utc).isoformat(), original_path=str(fixture),
            original_campaign_relative_path=str(fixture.relative_to(ROOT)), original_bytes=length,
            original_sha256=digest.hexdigest(), original_retained=True,
            compression=dict(codec="zlib", wbits=zlib.MAX_WBITS, level=6),
            zlib_version=zlib.ZLIB_VERSION, zlib_runtime_version=zlib.ZLIB_RUNTIME_VERSION,
            chunk_bytes_limit=chunk_bytes, compressed_stream_bytes=writer.total,
            compressed_stream_sha256=writer.stream_hash.hexdigest(), chunks=writer.rows,
            reconstruction=dict(argv=command, command=shlex.join(command), cwd="ANY",
                                output_policy="CREATE_NEW_ONLY"),
            semantics="Opaque byte stream. No tensor load, dtype conversion, or evidence filtering.")
        # Read back all chunks and decode before publishing a complete manifest.
        manifest["archive_readback"] = check_stream(directory, manifest)
        if file_hash(fixture) != (length, digest.hexdigest()) or identity(before) != identity(fixture.stat()):
            raise ValueError("The original fixture changed before archive publication.")
        manifest["original_post_archive_hash_matches"] = True
        with (directory / "manifest.json").open("x") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return manifest
    finally:
        writer.close_partial()


def verify_archive(manifest_path):
    path, manifest = read_manifest(manifest_path)
    return check_stream(path.parent, manifest)


def restore_archive(manifest_path, output):
    path, manifest = read_manifest(manifest_path)
    output = scoped_path(output)
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("The restored fixture requires a new path and an existing parent directory.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".fixture-restore-", suffix=".partial",
                                         dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            result = check_stream(path.parent, manifest, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # link() fails if output appeared concurrently. It never replaces an existing file.
        os.link(temporary, output)
        return dict(result, output=str(output), original_retained=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
