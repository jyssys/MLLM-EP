"""State phase decisions without turning admission relief into a target pass."""
from .common import *
from .reporting import FINAL, load, md


def main():
    verdict=load(ROOT/'final_per_model_verdict.json',{})
    work=load(ROOT/'state/final_work_audit.json',{})
    phase_rows=[]
    for model in ('ouro','huginn'):
        selected=verdict.get(model,{})
        c=selected.get('same_start_evidence') or {}
        profiles=[]
        for p in (ROOT/'raw').glob('*/profile_analysis.json'):
            cfg=load(ROOT/'configs'/(p.parent.name+'.json'),{})
            if cfg.get('model')!=model:continue
            v=json.loads(p.read_text())
            if v.get('eligible_for_matched_clean'):profiles.append(p.parent.name)
        phase_rows.extend([
            [model,'0: internal correctness','PASS' if work.get('passed') else 'REVIEW_REQUIRED',
                'Recorded token/state gates and full-depth audit only. Original-HF fidelity remains unmeasured.'],
            [model,'A: admission','CONDITIONAL',
                'Supported controls reduce specific binding limits. Long-context KV and slot limits remain workload dependent.'],
            [model,'B: profiling','PASS_SCOPED' if profiles else 'BLOCKED',
                f'{len(profiles)} matched profiles meet the predeclared wall-perturbation gate. Other profiles remain diagnostic.'],
            [model,'C: workload/QPS search','COMPLETED_TESTED_GRID',
                'Primary and secondary length sweeps remain visible. The chosen finite point has three paired starts when recorded.'],
            [model,'D: same-start target','NOT_CONFIRMED',
                f'P compute gate={c.get("P_compute_gate","NOT_MEASURED")}. D SLO gate={c.get("D_SLO_gate","NOT_MEASURED")}. D credit gate={c.get("D_credit_gate","NOT_MEASURED")}.'],
            [model,'E: actual D borrowing','NOT_RUN_PREREQUISITE',
                'No complete same-start compute/SLO/credit prerequisite supports the real-work probe. No synthetic point substitutes for it.'],
            [model,'F: artifacts','COMPLETE_AFTER_VERIFIED_PUSH',
                'The final response supplies the remote SHA. The publication receipt records the independent verification.']])
    text='\n## Phase-by-phase decisions\n\n'+md(['Model','Phase','Decision','Evidence scope'],phase_rows)+'\n\n'
    text+='## Why the target remains unconfirmed\n\n'
    text+='Ouro admission relief changes available reservations and transfer draining. It does not create additional HBM or prove GPU saturation.\n'
    text+='Huginn lighter inputs reduce per-request KV capacity. They also reduce Prefill compute, so the trade-off remains visible.\n'
    text+='Several starts have legal D memory and acceptable Decode tails. Their P device-activity or service/backlog gate still fails.\n'
    text+='Other starts have stronger pressure but binding D admission, a failed Decode tail, or a materially perturbed profile.\n'
    text+='These conditions cannot be combined across starts into a positive result.\n'
    text+='The measured kernel gaps are not exclusive CPU delay. A profiler does not identify the first historical blocked operation.\n'
    text+='The experiment therefore does not prove that recurrence itself prevents Prefill-heavy serving.\n'
    text+='A full Dynamic A campaign remains unjustified on this evidence.\n\n'
    text+='## Implemented changes and explicit non-changes\n\n'
    text+='All new source resides in `poc_flashvep/looped_pd_headroom_recovery/`.\n'
    text+='The implementation uses supported native memory/slot/wire-credit controls. Native release and ACK dependencies remain intact.\n'
    text+='The harness adds exact descriptor preparation, durable asynchronous evidence, and real fixed-depth graph warm-up.\n'
    text+='The native CUDA-only profiler and selected-stage launcher collect the device evidence.\n'
    text+='These controls are generic runtime changes, not a new Looped-specific scheduling contribution.\n'
    text+='The external qualified runtime and its working diff remain unchanged.\n'
    text+='No compact Huginn KV allocator, delayed-D-reserve protocol, new Dynamic A policy, or actual D borrowing path is implemented.\n'
    text+='No model weight, BF16 precision, fixed depth, or attention equation changes.\n'
    text+='LoopServing remains a read-only reference. The applicability matrix explains the API and execution differences.\n'
    (ROOT/'phase_decisions.md').write_text('# Phase decisions\n'+text)
    FINAL.write_text(FINAL.read_text()+text)
    write(ROOT/'state/phase_decisions.json',dict(time=time.time(),rows=phase_rows))
    print(json.dumps(dict(phase_rows=len(phase_rows))))


if __name__=='__main__':main()
