"""Change dispatch priority, not the frozen full200 experiment settings."""
import time
from .common import ROOT,write
from .full200_confirmations import main as confirm


def main():
    write(ROOT/'configs/full200_dispatch_priority_amendment_v1.json',dict(time=time.time(),
        reason='The user added P0 historical reproduction and causal bridge work. Protect the predeclared full200 independent repeats within the unchanged budget.',
        change='Start the same full200 confirmation cells before the optional pilot grid finishes.',
        unchanged=['All dataset IDs and hashes','All arrival seeds and offsets','Cuts2/4/8 and prior cut12 repeats',
            'BI0,actual1/1/1,c4,h4,KV128,B8192/S1024','SLO and confidence-interval method','All failed/unfavorable results'],
        scope='No active GPU job is stopped. A waiting controller is replaced at an audited no-child checkpoint. No pilot completion flag is fabricated.'))
    confirm(wait_for_pilot=False)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/full200_allcuts_confirmation_hold.json',dict(time=time.time(),error=repr(error)));raise
