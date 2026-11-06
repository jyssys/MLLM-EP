"""Check current collector plumbing. Do not promote it to serving evidence."""
import copy
from .common import *


def main():
    initialize()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from .graph_activity import build
    with quiet_analysis('current_collector_source_build'):build()
    cfg=copy.deepcopy(json.loads((ROOT/'configs/collector_device1_flush1_launches10000_v2.json').read_text()))
    cfg.update(name='collector_device1_flush1_launches10000_v3_invalid_dump',
        phase='COLLECTOR_SOURCE_BUILD_AND_PLUMBING_ONLY_NOT_SAFE_SLACK')
    save(cfg)
    from .run import execute
    complete=execute(cfg['name'])
    write(ROOT/'state/current_collector_build_check.json',dict(name=cfg['name'],complete=complete,
        scope='Synthetic collector plumbing and timestamp/export assertions only. No model, serving tail, Safe Slack, or valid future budget is established.'))
    assert complete


if __name__=='__main__':main()
