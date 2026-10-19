"""Reuse the pinned runtime. Keep the preceding evidence unchanged."""
from poc_decode_first_depth_aware_dp_v3.contract import *

V3_ROOT = ROOT
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
BASE_COMMIT = '0eb9c04d1eef6af33c30a0ca2c0244fb26e41950'
SPEC = Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/depth_cdb_factorial_hindsight_spec.md')
ARMS = {
    'A0': dict(label='Naive Ouro (w/ dynamic depth) GPU1-DP1', gpus=[4], refill=False, hindsight=False),
    'A1': dict(label='Naive DP4 Ouro (w/ dynamic depth)', gpus=GPUS, refill=False, hindsight=False),
    'A2': dict(label='Naive DP4 Ouro (w/ dynamic depth) + Hindsight', gpus=GPUS, refill=False, hindsight=True),
    'A3': dict(label='DP4 Ouro + CDB', gpus=GPUS, refill=True, hindsight=False),
    'A4': dict(label='DP4 Ouro + CDB + Hindsight', gpus=GPUS, refill=True, hindsight=True),
}

def lpt(ids, work, ranks=4):
    """Clairvoyant depth-sum LPT. Never use correctness labels."""
    loads = [0.0] * ranks
    counts = [0] * ranks
    result = {}
    order = {rid: i for i, rid in enumerate(ids)}
    for rid in sorted(ids, key=lambda rid: (-work[rid], order[rid])):
        rank = min(range(ranks), key=lambda r: (loads[r], counts[r], r))
        result[rid] = rank
        loads[rank] += work[rid]
        counts[rank] += 1
    return result
