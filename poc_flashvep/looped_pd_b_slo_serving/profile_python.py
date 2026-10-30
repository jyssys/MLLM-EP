#!/usr/bin/env python3
"""Diagnostic launcher. Enable the existing native RLT profiler after warmup."""
import os
import sys

if sys.path and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)

from poc_flashvep.looped_pd_abc import worker as reused
from poc_flashvep.looped_pd_b_slo_serving import worker


def main():
    if sys.argv[1:3] != ['-m', 'poc_flashvep.looped_pd_b_slo_serving.worker'] or len(sys.argv) != 4:
        raise ValueError('This launcher accepts only the registered follow-up worker and one configuration.')
    original_main = reused.main

    def profiled_main():
        sys.argv.append('--profile')
        return original_main()

    reused.main = profiled_main
    sys.argv = sys.argv[2:]
    worker.main()


if __name__ == '__main__':
    main()
