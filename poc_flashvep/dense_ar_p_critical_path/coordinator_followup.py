"""Test coordinator GC separately from evidence-write deferral."""
import json
import time
from .common import ROOT,write,bind,digest
from .bootstrap import baseline,launch
from .capsule import build as retirement_source
from .coordinator_capsule import build


def prepare():
    runner=bind();names=[];cases=[]
    for label in ('Hybrid','StockToken','NativeStrong'):
        jobs=[]
        for deferred_gc in (False,True):
            name=f'PC_CoordGC_{label}_{"DeferredGC" if deferred_gc else "NormalGC"}_BufferedReplay192'
            cfg=baseline(label,name,[2,3],seed=2026112322,save_cfg=False)
            parent=retirement_source(cfg['source_capsule'].removeprefix('pc_'))
            source=build(parent.name,'r2')
            cfg.update(source_capsule=source.name,source_capsule_SHA256=digest(source),
                correctness_only=True,axis_state_capture=False,state_capture=False,gpu_state_digest=False,
                P_critical_trace=True,P_event_timing=False,source_credit_release_on_retirement=False,
                coordinator_trace=True,buffer_coordinator_evidence=True,defer_coordinator_cyclic_gc=deferred_gc,
                evidence='DIAGNOSTIC',scope='Both arms defer evidence writes. Only coordinator cyclic-GC timing changes. '
                    'All workers remain byte-identical. Preserve actual client times and all output work. '
                    'No primary performance claim follows before fresh qualification and independent confirmation.')
            cfg.pop('qualification_gate_v2',None);runner.save(cfg);names.append(name);jobs.append(name)
        cases.append(dict(label=label,jobs=jobs,seed=2026112322,pair=[2,3]))
    path=ROOT/'jobs/coordinator_GC_diagnostic_v1.json';assert not path.exists()
    write(path,dict(time=time.time(),lanes=[[],names,[],[]],cases=cases,stop_on_failure=True,
        evidence='DIAGNOSTIC',
        observation='A Hybrid immediate-write loop gap of79.30ms contained78.87ms in a file write. '
            'After write deferral,a37.48ms loop gap contained35.17ms of cyclic GC. '
            'A separate Stock loop gap contained37.26ms of GC. These are host enclosures,not GPU work.',
        hypothesis='If coordinator cyclic GC causes delivery tails,deferring GC will remove those overlaps without changing worker gaps.',
        falsifier='Persistent large client gaps outside GC or unchanged worker tails contradict GC as the sole cause.',
        scope='One-factor diagnostic after the write-deferral experiment. No artificial sleep or timestamp correction. '
            'Restore the caller GC state after measurement and on exception. Record coordinator maximum RSS. '
            'Do not call a common coordinator improvement a Layer-specific kernel gain.'))
    return launch(path.name)


if __name__=='__main__':print(prepare(),flush=True)
