"""Isolated startup hook used only by the DeepEP revalidation commands."""

from __future__ import annotations

import os


if os.environ.get("FLASHVEP_DEEPEP_PROOF_DIR"):
    from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe

    install_backend_probe()

if os.environ.get("FLASHVEP_DEEPEP_REPLAY_RESULT_DIR"):
    from poc_flashvep.deepep_revalidation.operator_replay import install_operator_replay

    install_operator_replay()

if os.environ.get("FLASHVEP_DBO_CORRECTNESS_FIX") == "1":
    from poc_flashvep.deepep_revalidation.dbo_correctness_probe import (
        install_dbo_correctness_fix,
    )

    install_dbo_correctness_fix()

if os.environ.get("FLASHVEP_DBO_CORRECTNESS_TRACE_DIR"):
    from poc_flashvep.deepep_revalidation.dbo_correctness_probe import (
        install_dbo_correctness_probe,
    )

    install_dbo_correctness_probe()
