"""Qualify an existing byte-exact optimization before using it in Native."""
import json
from .common import ROOT,save,write
from .qualification import configuration,records
from .run import execute


def main():
    name='qual_Native_COALESCED_LongBench_N8_B8192_S1024_HND_BI1_v1'
    cfg=configuration(name,'Native','qualification8.json',batch_invariant=True,nixl_manual_coalesce=True)
    assert execute(save(cfg)),name
    folder=ROOT/'raw'/name
    teacher=ROOT/'raw/qual_Standard_LongBench_N8_B8192_S1024_HND_BI1_v1'
    outputs={r['prompt_id']:r for r in records(teacher/'requests.jsonl')}
    state=json.loads((teacher/'prompt_state_single.json').read_text())['requests']
    P=json.loads((folder/'prompt_state_prefill.json').read_text())['requests']
    D=json.loads((folder/'prompt_state_decode.json').read_text())['requests']
    checks=[]
    for r in records(folder/'requests.jsonl'):
        rid=f'q{r["sample_id"]}';expected=state[rid]
        checks.append(dict(request_id=rid,output=r['output_ids']==outputs[r['prompt_id']]['output_ids'],
            P_KV=P[rid]['KV']==expected['KV'],P_hidden=P[rid]['final_hidden_SHA256']==expected['final_hidden_SHA256'],
            D_KV=D[rid]['KV']==expected['KV']))
    tx=json.loads((folder/'NIXL_host_setup_decode.json').read_text())['rows']
    merged=[r for r in tx if r['kind']=='make_transfer_host' and r['submitted_descriptors']<r['input_descriptors']]
    lifetime={p.name:json.loads(p.read_text())['passed'] for p in folder.glob('KV_lifetime_*.json')}
    passed=len(checks)==8 and all(all(v for k,v in r.items() if k!='request_id') for r in checks) and bool(merged) and len(lifetime)==2 and all(lifetime.values())
    write(ROOT/'analysis/coalesced_Native_N8_full_state_gate.json',dict(passed=passed,checks=checks,lifetime=lifetime,
        coalesced_transactions=len(merged),evidence='CORRECTNESS_ONLY',
        scope='Exact registered-byte range merging. Unchanged Native model, source/destination ownership, ACK and scheduler. Not a Layer/Token method. CPU witnesses invalidate performance timings.'))
    print('COALESCED_FULL_STATE_GATE',passed,flush=True)


if __name__=='__main__':main()
