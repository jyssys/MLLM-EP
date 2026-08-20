"""Opt-in startup hook for tile-to-slack DeepEP replay."""

from __future__ import annotations

import os


if os.environ.get("FLASHVEP_TILE_REPLAY_RESULT_DIR"):
    from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe
    from poc_flashvep.tile_slack_mechanism.operator_replay import (
        install_tile_slack_replay,
    )

    install_backend_probe()
    install_tile_slack_replay()
