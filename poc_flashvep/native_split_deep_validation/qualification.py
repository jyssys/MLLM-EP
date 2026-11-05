"""Reuse every existing exactness gate with the new immutable dispatcher."""
from .common import initialize
from .run import execute


def qualify(name, reference=None):
    initialize()
    from ..native_loop_2d_oracle import qualification as inherited
    inherited.initialize = initialize
    inherited.execute = execute
    return inherited.qualify(name, reference)
