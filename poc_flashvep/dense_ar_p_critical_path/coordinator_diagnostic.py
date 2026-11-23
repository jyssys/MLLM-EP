"""Test a measured client-delivery stall without calling it GPU compute."""
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
        for buffered in (False,True):
            name=f'PC_CoordDiagnostic_{label}_{"DeferredIO" if buffered else "ImmediateIO"}_Replay192'
            cfg=baseline(label,name,[2,3],seed=2026112322,save_cfg=False)
            parent=retirement_source(cfg['source_capsule'].removeprefix('pc_'))
            source=build(parent.name)
            cfg.update(source_capsule=source.name,source_capsule_SHA256=digest(source),
                correctness_only=True,axis_state_capture=False,state_capture=False,gpu_state_digest=False,
                P_critical_trace=True,P_event_timing=False,source_credit_release_on_retirement=False,
                coordinator_trace=True,buffer_coordinator_evidence=buffered,evidence='DIAGNOSTIC',
                scope='Diagnostic only. Same actual request/arrival/output contract as the observed long client-delivery stall. '
                    'All worker bytes are identical to the old source. Only coordinator evidence I/O and optional host observation differ. '
                    'This does not qualify a serving optimization before fresh correctness and independent confirmation.')
            cfg.pop('qualification_gate_v2',None);runner.save(cfg);names.append(name);jobs.append(name)
        cases.append(dict(label=label,jobs=jobs,seed=2026112322,pair=[2,3]))
    path=ROOT/'jobs/coordinator_delivery_io_diagnostic_v1.json';assert not path.exists()
    write(path,dict(time=time.time(),lanes=[[],names,[],[]],cases=cases,stop_on_failure=True,
        hypotheses=['Evidence-file writes block the coordinator.','GC or JSON work blocks the coordinator.',
            'A partial socket receive blocks the coordinator.','Other host scheduling or shared-resource interference.'],
        existing_sender='Already asynchronous. Do not implement another asynchronous sender without evidence.',
        discriminating_test='Trace append serialization/write,GC,receive,slow poll and process/thread CPU during loop gaps. '
            'One-factor defer evidence writes across Hybrid,Stock Token and Native. No sleep or artificial serving delay.',
        scope='No winner claim from diagnostic timings. Preserve the uninstrumented long-tail run. '
            'Buffering evidence is benchmark/runtime hygiene,not layer-specific GPU efficiency. '
            'A host overlap is not causal proof. Reject the I/O hypothesis if the stall occurs outside writes.'))
    return launch(path.name)


if __name__=='__main__':print(prepare(),flush=True)
