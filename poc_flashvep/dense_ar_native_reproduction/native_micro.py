"""Compare the same clean decode contract on standard Single-GPU and Native PD."""
import json
import time
import hashlib
from .common import ROOT,write,save
from .qualification import configuration,records
from .run import execute
from .metrics import summarize


def main():
    gate=ROOT/'analysis/coalesced_Native_N8_full_state_gate.json'
    while not gate.exists():time.sleep(2)
    assert json.loads(gate.read_text())['passed']
    data=json.loads((ROOT/'data/diagnostic_L8192.json').read_text())[:8]
    for row in data:
        row.update(input_tokens=8192,task='DIAGNOSTIC_REAL_DOCUMENT',
            input_ids_SHA256=hashlib.sha256(json.dumps(row['input_ids']['qwen8'],separators=(',',':')).encode()).hexdigest())
    path=ROOT/'data/diagnostic_serving_L8192.json'
    if path.exists():assert json.loads(path.read_text())==data
    else:write(path,data)
    values=[];teacher=None
    for label,extra in (('Standard',dict(reference_single=True,gpus=[1])),
        ('OfficialPD',dict(nixl_manual_coalesce=False)),('CoalescedPD',dict(nixl_manual_coalesce=True))):
        name=f'micro_{label}_L8192_O16_N8_C1_BI1_B8192_S1024_v1'
        cfg=configuration(name,'Native',path.name,batch_invariant=True,output_cap=16,ignore_eos=True,
            state_capture=False,correctness_only=False,diagnostic_only=True,**extra)
        assert execute(save(cfg)),name
        actual={r['prompt_id']:r for r in records(ROOT/'raw'/name/'requests.jsonl')}
        if teacher is None:teacher=actual
        checks={rid:r['output_ids']==teacher[rid]['output_ids'] for rid,r in actual.items()}
        metrics=summarize(name)
        values.append(dict(name=name,strict_output_parity=len(checks)==8 and all(checks.values()),checks=checks,
            metrics={k:v for k,v in metrics.items() if k not in ('per_request','by_task')},
            evidence='MEASURED_DIAGNOSTIC',scope='Same GPU1 decode, stock execution, official invariant numerical path, concurrency1 and fixed16 diagnostic. Single-GPU is not a two-GPU throughput comparator.'))
        write(ROOT/'analysis/clean_Native_standard_decode_comparison_v1.json',dict(rows=values))
    assert all(r['strict_output_parity'] for r in values)


if __name__=='__main__':main()
