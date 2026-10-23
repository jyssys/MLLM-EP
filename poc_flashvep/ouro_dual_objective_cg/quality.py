"""Reuse frozen public prompts, evaluator stops, and full-answer generation."""
import json
from .common import ROOT,write
from .adaptive_selector import DualControl
from .work_ledger import WorkLedger

def run_job(model,job,target):
    from ..ouro_budget_joint_cg import quality as old
    old.ROOT=ROOT
    cells={c['label']:c for c in job['cells']}
    norm=json.loads((ROOT/'objective_normalization_v1.json').read_text())
    original_engine=old.engine
    engine_index=0
    def measured_engine(*args,**kwargs):
        nonlocal engine_index
        label=job['cells'][engine_index]['label'];engine_index+=1
        e=original_engine(*args,**kwargs);ledger=WorkLedger(e);native_close=e.close
        def close():
            from collections import defaultdict
            work=defaultdict(int)
            for row in ledger.rows:
                for span in row['spans']:work[span['id']]+=span['tokens']*span['G']
            expected={r['id']:4*len(r['input_ids']) for r in json.loads((ROOT/'quality_manifest.json').read_text())['rows'] if r['id'] in job['ids']}
            valid=all(work[rid]==count for rid,count in expected.items())
            write(target/f'{label}_prefill_work.json',dict(policy=label,actual_work=dict(work),expected_work=expected,full_R4_prompt=valid,
                scope='Native CPU submission ledger, checked after all free-running quality requests finish.'))
            ledger.close();e.close=native_close;native_close()
            assert valid,'Quarantine quality arm: prompt Full-R4 work mismatch'
        e.close=close
        return e
    old.engine=measured_engine
    class QualityControl(DualControl):
        def __init__(self,engine,lookup,policy_id,*unused):
            self.label=policy_id;self.dump_index=0
            super().__init__(engine,lookup,norm,cells[policy_id])

        def dump(self):
            if self.selector.trace:
                write(target/f'{self.label}_decision_cohort_{self.dump_index}.json',dict(
                    policy=self.label,optimizer_decisions=self.selector.trace,chunk_decisions=self.actions.decisions,
                    output_control_events=self.output_events,
                    scope='QUALITY_DIAGNOSTIC. Includes the preserved detokenization/stop-check client. Not clean serving.'))
                self.dump_index+=1

        def reset_cohort(self):
            self.dump();self.reset()

        def close(self):
            self.dump();super().close()
    old.OnlineControl=QualityControl
    return old.run_job(model,job,target)
