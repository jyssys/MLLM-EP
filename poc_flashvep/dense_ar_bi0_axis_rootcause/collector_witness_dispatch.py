"""Run the collector witness at the next approved quiet boundary."""
import time
from .common import ROOT,write,save,initialize
from .run import execute


def main():
    initialize()
    cfg=dict(name='CorrelatedCollector_Witness_v2',method='Native',
        entry_module='poc_flashvep.dense_ar_bi0_axis_rootcause.correlation_witness',
        gpus=[1],batch_invariant=False,correctness_only=True,wall_timeout=180)
    passed=execute(save(cfg))
    write(ROOT/'state/correlated_collector_witness_complete.json',dict(time=time.time(),passed=passed))


if __name__=='__main__':main()
