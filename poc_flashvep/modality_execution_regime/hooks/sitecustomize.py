"""Opt-in hook for the modality execution-regime replay."""

import os

if os.environ.get("FLASHVEP_MODALITY_REPLAY_DIR"):
    from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe
    from poc_flashvep.modality_execution_regime.operator_replay import install

    install_backend_probe()
    install()
