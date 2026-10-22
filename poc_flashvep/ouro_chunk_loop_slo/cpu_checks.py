"""Upstream lifecycle tests. Tiny CPU fixtures are not model-quality evidence."""

import os
import subprocess
import time
from .common import ROOT, FORK, PYTHON, write


def main():
    command = [
        PYTHON,
        "-m",
        "pytest",
        "-q",
        "tests/test_split_prefill.py",
        "--disable-warnings",
    ]
    started = time.time()
    result = subprocess.run(
        command,
        cwd=FORK,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(FORK)),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    write(
        ROOT / "cpu_lifecycle_tests.json",
        dict(
            command=command,
            cwd=str(FORK),
            CUDA_VISIBLE_DEVICES="",
            return_code=result.returncode,
            started=started,
            ended=time.time(),
            output=result.stdout,
            evidence="SOFTWARE_UNIT_TEST_ONLY_TINY_CPU_FIXTURES_NOT_PRETRAINED_QUALITY_OR_GPU_TIMING",
        ),
    )
    print(result.stdout)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
