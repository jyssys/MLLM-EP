"""Separate batch invariance from PD transport before changing Native."""
import json
import time
from .common import ROOT,PREVIOUS,write,digest,save,initialize
from .run import execute


def main():
    initialize()
    for length in (4096,8192,16384):
        source=PREVIOUS/'data'/f'discovery36_L{length}.json'
        target=ROOT/'data'/f'diagnostic_L{length}.json'
        rows=json.loads(source.read_text())[:8]
        if target.exists():assert json.loads(target.read_text())==rows
        else:write(target,rows)
    declaration=ROOT/'configs/micro_reference_design_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),model_revision='b968826d9c46dd6066d109eabc6255188de91218',
            lengths=[4096,8192,16384],output=16,concurrency=1,GPU=1,
            hypotheses=[
                'H1: Batch-invariant kernel selection explains a large part of decode latency. Changing only that flag changes the standard-engine time.',
                'H2: Graph/Eager or compile settings dominate. The flag-only comparison is small but stock execution defaults differ.',
                'H3: Custom PD host/control or transfer dominates. The same standard-engine numerical path remains materially faster than Custom Native.'],
            methods=['standard_matched_BI0','standard_matched_BI1','standard_stock_BI0'],
            actual_B_S=[8192,1024],KV_GiB=42,warmup_per_length=2,measured_per_length=6,
            boundary_sync='Matched BI0/BI1 synchronize only before request, at first sample, and after final output. Stock default omits the first-sample barrier.',
            scope='Fixed lengths are diagnostic controls only. No Layer/Token measurement or QPS sweep. Stock-default differences are not a flag-only causal comparison.'))
    for label,bi,stock in (('standard_matched_BI0',False,False),('standard_matched_BI1',True,False),('standard_stock_BI0',False,True)):
        name=f'{label}_L4096_8192_16384_O16_C1_v2'
        cfg=dict(name=name,method='Native',entry_module='poc_flashvep.dense_ar_native_reproduction.single_gpu',
            gpus=[1],batch_invariant=bi,stock_execution_defaults=stock,boundary_sync=not stock,
            diagnostic_only=True,wall_timeout=1200)
        assert execute(save(cfg)),name
    write(ROOT/'state/standard_micro_reference_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':main()
