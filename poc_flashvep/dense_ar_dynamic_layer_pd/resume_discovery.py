"""Resume the incomplete output sweep after independent SLO calibration."""
from .common import *


def main():
    initialize();started=time.monotonic()
    while not (ROOT/'configs/frozen_SLO_regimes.json').exists():
        assert time.monotonic()-started<3600,'SLO calibration did not finish'
        time.sleep(5)
    from .discovery import main as discover
    discover()


if __name__=='__main__':main()
