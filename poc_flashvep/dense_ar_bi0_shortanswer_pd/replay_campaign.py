"""Wait for independent costs and measured controls before oracle analysis."""
import time
from .common import ROOT,write


def main(fast=False):
    version='v2' if fast else 'v1';discovery_version='v3' if fast else 'v2'
    path=ROOT/'configs'/f'frozen_conditional_replay_{version}.json'
    if not path.exists():
        write(path,dict(time=time.time(),calibration='cost_calibration20.json',validation='cost_validation15.json',
            model='Nonnegative intercept + L/8192 + (L/8192)^2, independently per action/service.',
            fixed_cuts=[1,2,4,6,8,12],fractions=[0,.25,.5,.75,1],source_credit=4,actual_batch=[1,1,1],
            QPS=[1.2,1.65,2.35],discovery_seed=2026112061,heldout_seeds=[2026112062,2026112063],
            output_contract='Fixed C1 Natural reference lengths inside the model only. No measured output is overwritten.',
            beam_width=24,oracle_future_information=True,
            qualification='The model lacks client ITL tails. TTFT-qualified token proxy cannot pass the full joint-SLO/online-axis gate.',
            scope='Conditional service/queue prediction, updated FIFO queues and source lifetime credit. '
            'A finite beam is not a global upper bound. Discovery-frozen and per-cell hindsight fixed choices are separate.'))
    for marker in ('continuation_costs_complete.json',f'controlled_fixed_axis_discovery_{discovery_version}_complete.json'):
        while not (ROOT/'state'/marker).exists():
            assert not (ROOT/'state/STOP_REQUESTED.json').exists()
            for failed in ('continuation_costs_hold.json','controlled_discovery_hold.json'):
                assert not (ROOT/'state'/failed).exists(),failed
            time.sleep(5)
    from .offline_headroom import main as analyze
    analyze();write(ROOT/'state/offline_conditional_replay_complete.json',dict(time=time.time(),complete=True,
        full_SLO_or_online_axis_gate=False))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--fast',action='store_true');args=parser.parse_args()
    try:main(args.fast)
    except BaseException as error:
        write(ROOT/'state/replay_campaign_hold.json',dict(time=time.time(),error=repr(error)));raise
