"""Preserve the startup-collision failure and rerun in a new namespace."""
import time
from .common import ROOT,write
from .baseline_v2 import main


if __name__=='__main__':
    try:main(tag='v3')
    except BaseException as error:
        write(ROOT/'state/baseline_v3_hold.json',dict(time=time.time(),error=repr(error)));raise
