"""Measure P-service device activity without D reservation or NIXL progress."""
from .common import *


def main():
    import torch
    from ..looped_pd_bottleneck_crossover import capacity_worker as inherited
    from . import capacity_worker
    cfg=config();original=inherited.cohort;folder=ROOT/'raw'/cfg['name']
    def cohort(engine,tokens,cap,seed,tag,barrier,recorder,use_barrier=None):
        timed=use_barrier is None and '_r' in tag
        if timed:
            marker='APD_P_DUTY|'+tag
            torch.cuda.nvtx.range_push(marker);anchor=time.monotonic()
        try:value=original(engine,tokens,cap,seed,tag,barrier,recorder,use_barrier)
        finally:
            if timed:torch.cuda.nvtx.range_pop()
        if timed:
            append(folder/'p_duty_anchors.jsonl',dict(marker=marker,pid=os.getpid(),anchor_monotonic=anchor,
                P_interval=value['intervals']['prefill'],batch=len(tokens),tag=tag,
                full_depth_rows=value['full_depth_rows'],observed_full_depth_rows=value['full_depth_rows_observed'],
                scope='Actual local StageEngine P interval. No D cache reservation, transfer, or synthetic KV. Coda and output delivery remain outside this P interval.'))
        return value
    inherited.cohort=cohort;capacity_worker.main()


if __name__=='__main__':main()
