"""Test first-sample reuse with the exact producer token-count bound."""
import json
from .common import ROOT,write,save
from .qualification import configuration,records
from .run import execute


def main():
    name='qual_Native_BOUNDED_LongBench_N8_B8192_S1024_HND_BI0_v1'
    cfg=configuration(name,'Native','qualification8.json',protocol='FIRST_TOKEN_REUSE_BOUNDED')
    assert execute(save(cfg)),name
    base=ROOT/'raw';gold={r['prompt_id']:r for r in json.loads((base/'standard_HND_BI0_LongBench200_natural_C1_cap2048_v1/natural_reference.json').read_text())['rows']}
    rows=records(base/name/'requests.jsonl');reference=json.loads((base/'qual_Standard_LongBench_N8_B8192_S1024_HND_BI0_v2/prompt_state_single.json').read_text())['requests']
    checks=[]
    P=json.loads((base/name/'prompt_state_prefill.json').read_text())['requests']
    D=json.loads((base/name/'prompt_state_decode.json').read_text())['requests']
    for row in rows:
        rid=f'q{row["sample_id"]}';expected=reference[rid]
        checks.append(dict(request_id=rid,prompt_id=row['prompt_id'],output=row['output_ids']==gold[row['prompt_id']]['output_ids'],
            P_KV=P[rid]['KV']==expected['KV'],P_hidden=P[rid]['final_hidden_SHA256']==expected['final_hidden_SHA256'],
            D_import_KV=D[rid]['KV']==expected['KV'],D_first_query=D[rid]['first_observed_query'],
            correct_D_first_query=D[rid]['first_observed_query']==row['input_tokens']))
    lifetime={p.name:json.loads(p.read_text())['passed'] for p in (base/name).glob('KV_lifetime_*.json')}
    passed=len(checks)==8 and all(all(row[k] for k in ('output','P_KV','P_hidden','D_import_KV','correct_D_first_query')) for row in checks) and all(lifetime.values()) and len(lifetime)==2
    write(ROOT/'analysis/Native_first_sample_reuse_bounded_N8_gate.json',dict(passed=passed,checks=checks,lifetime=lifetime,
        evidence='CORRECTNESS_ONLY',scope='Explicit remote-token upper bound adapter. Unchanged model math. CPU state copies invalidate all timing comparisons. Not an unmodified official proxy run.'))
    print('BOUNDED_NATIVE_GATE',passed,flush=True)


if __name__=='__main__':main()
