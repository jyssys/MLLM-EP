"""Isolated startup hook used only by the DeepEP revalidation commands."""

from __future__ import annotations

import os


if os.environ.get("FLASHVEP_DEEPEP_PROOF_DIR"):
    from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe

    install_backend_probe()

if os.environ.get("FLASHVEP_DEEPEP_REPLAY_RESULT_DIR"):
    from poc_flashvep.deepep_revalidation.operator_replay import install_operator_replay

    install_operator_replay()
