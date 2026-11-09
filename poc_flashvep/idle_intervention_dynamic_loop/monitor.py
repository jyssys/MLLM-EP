"""Retain progress checks and burner recovery through the publication window."""
from .common import *


if __name__=='__main__':
    configure()
    from ..adaptive_idle_loop_continuation import monitor
    monitor.configure=configure;monitor.ROOT=ROOT
    # The independent GPU guard still stops experiments at GPU_DEADLINE.
    # Idle resource recovery must remain available after that stop boundary.
    monitor.GPU_DEADLINE=DEADLINE
    monitor.main()
