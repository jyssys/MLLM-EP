"""Start the next registered stage after the owned stage and its cleanup exit."""

import argparse
from pathlib import Path
import subprocess
import time
from .common import REPO, ROOT, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid", required=True, type=int)
    p.add_argument("--manifest", required=True, nargs="+")
    a = p.parse_args()
    process = Path(f"/proc/{a.pid}/cmdline")
    while process.exists():
        command = process.read_bytes().replace(b"\0", b" ")
        if b"ouro_chunk_loop_slo/run_manifest.sh" not in command:
            raise RuntimeError("Expected owned stage process changed")
        write(
            ROOT / "next_stage_status.json",
            dict(
                status="WAITING_FOR_OWNED_STAGE_AND_CLEANUP",
                pid=a.pid,
                manifest=a.manifest,
            ),
        )
        time.sleep(3)
    write(
        ROOT / "next_stage_status.json",
        dict(status="RUNNING_NEXT_STAGE", manifest=a.manifest),
    )
    code = 0
    for manifest in a.manifest:
        code = subprocess.call(
            ["bash", "poc_flashvep/ouro_chunk_loop_slo/run_manifest.sh", manifest],
            cwd=REPO,
        )
        if code:
            break
    write(
        ROOT / "next_stage_status.json",
        dict(
            status="COMPLETE" if code == 0 else "FAILED",
            exit_code=code,
            manifest=a.manifest,
        ),
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
