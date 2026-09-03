"""Run one direct replay rank under Nsight Systems (called by torchrun)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

out = Path(os.environ["FLASHVEP_ATLAS_RESULT_DIR"])
rank = int(os.environ["LOCAL_RANK"])
base = out / ("resource_atlas_main" if rank == 0 else f"resource_atlas_rank{rank}")
cmd = [
    "nsys", "profile", "--trace=cuda,osrt", "--sample=none",
    "--cpuctxsw=none", "--backtrace=none", "--stats=false",
    "--export=sqlite", "--force-overwrite=true", "-o", str(base),
    os.environ["FLASHVEP_PYTHON"], str(Path(__file__).with_name("replay_rank.py")),
]
(out / f"replay_nsys_command_rank{rank}.txt").write_text(" ".join(cmd) + "\n")
raise SystemExit(subprocess.run(cmd, env=os.environ.copy()).returncode)
