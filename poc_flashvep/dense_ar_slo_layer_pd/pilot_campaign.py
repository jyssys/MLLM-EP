"""Run the predeclared pilot stages and preserve the stopping condition."""
from .common import *
from .frontier import run
from .stages import tuning,qualify,duration_initial


def main():
    initialize()
    try:
        write(ROOT/'state/pilot_campaign.json',dict(time=time.time(),stage='COLOCATED_SLO_FRONTIER',status='RUNNING'))
        run(tuning())
        write(ROOT/'state/pilot_campaign.json',dict(time=time.time(),stage='STATE_AND_DIRECT_DEADLINE_QUALIFICATION',status='RUNNING'))
        qualify()
        write(ROOT/'state/pilot_campaign.json',dict(time=time.time(),stage='FIXED_WINDOW_LOW_LOAD_FOUR_METHOD_SCREEN',status='RUNNING'))
        duration_initial()
        write(ROOT/'state/pilot_campaign.json',dict(time=time.time(),stage='PILOTS_COMPLETE_REVIEW_BEFORE_EXPANSION',status='COMPLETE'))
    except BaseException as error:
        write(ROOT/'state/pilot_campaign.json',dict(time=time.time(),status='STOPPED_FOR_REVIEW',error=repr(error)))
        raise


if __name__=='__main__':main()
