"""Test actual mixed decisions with four declared correctness arrival clusters."""
import copy
from .common import *


def main():
    initialize()
    source='r9_ouro_actual_mixed_state32'
    cfg=copy.deepcopy(json.loads((ROOT/'configs'/(source+'.json')).read_text()))
    cfg.update(name='r10_ouro_actual_mixed_state32',
        arrival_offsets=[6.*(index//8) for index in range(32)],arrival_rate=None,
        arrival_seed=None,arrival_pattern='FOUR_EIGHT_REQUEST_CORRECTNESS_CLUSTERS',
        phase='REAL_ONLINE_MIXED_CUT_CORRECTNESS_STRESS_NOT_SERVING_PERFORMANCE',
        stress_scope='Four clusters change actual Decode occupancy. No cut is assigned. All cuts require fresh D decisions. State-observer timing is not performance evidence.')
    save(cfg)
    path=ROOT/'state/queue.json'
    queue=json.loads(path.read_text())
    assert cfg['name'] not in queue
    queue.append(cfg['name'])
    write(path,queue)


if __name__=='__main__':main()
