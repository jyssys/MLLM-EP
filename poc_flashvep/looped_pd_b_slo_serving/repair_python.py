#!/usr/bin/env python3
"""Apply a diagnostic variant after Python completes site initialization."""
import os
import sys

if sys.path and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)  # Do not let this package's profile.py shadow Python's stdlib profile module.

from poc_flashvep.looped_pd_b_slo_serving.repair_hooks import install

# Multiprocessing executes this module as __mp_main__ before it creates an engine.
install()
from poc_flashvep.looped_pd_b_slo_serving.repair_capture import install as capture
capture()


def main():
    if sys.argv[1:3] != ['-m', 'poc_flashvep.looped_pd_b_slo_serving.worker'] or len(sys.argv) != 4:
        raise ValueError('Expected the registered follow-up worker and one configuration.')
    from poc_flashvep.looped_pd_b_slo_serving import worker
    sys.argv = sys.argv[2:]
    worker.main()


if __name__ == '__main__':
    main()
