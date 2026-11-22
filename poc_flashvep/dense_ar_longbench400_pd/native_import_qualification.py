"""Check Native valid-prefix transport without hashing an uncomputed last slot."""
import json
from .common import ROOT,write
from .experiment import create
from .freeze_capsule import freeze
from .run import execute
from .qualification import intervals_cover


def main():
    capsule,sha=freeze('native_valid_prefix_observer_qualification_v2_20261121')
    name=create('Q22_NativeValidPrefix_N8_FullState_v2','Native','qualification8.json',
        pair=(6,7),correctness=True,controlled=True,serial=True,output=16,fixed=True,
        execution_family='INSTALLED_DEFAULT_COMPILE',state_capture=True,axis_state_capture=False,
        gpu_state_digest=True,source_capsule=capsule,source_capsule_SHA256=sha,
        wall_timeout=3600,serving_timeout=3300,
        scope='Read-only Native transport observer. Before D execution,compare only the valid L-1 prefix. '
            'After the original last-prompt query executes,capture the complete prompt KV and hidden. '
            'Preserve earlier full-L-before-query observations with their limited validity. No timing claim.')
    assert execute(name),name
    folder=ROOT/'raw'/name
    source=json.loads((folder/'native_valid_prefix_prefill.json').read_text())['requests']
    dest=json.loads((folder/'native_valid_prefix_decode.json').read_text())['requests']
    full=json.loads((folder/'prompt_state_decode.json').read_text())['requests']
    data=json.loads((ROOT/'data/qualification8.json').read_text());checks=[]
    queries={role:json.loads((folder/f'actual_queries_{role}.json').read_text())['rows'] for role in ('prefill','decode')}
    for i,row in enumerate(data):
        rid='q'+str(i);length=row['input_tokens']
        assert source[rid]['valid_prefix_tokens']==dest[rid]['valid_prefix_tokens']==length-1
        assert source[rid]['KV']==dest[rid]['KV'],(rid,'Native valid imported prefix differs')
        assert len(full[rid]['KV'])==36 and full[rid]['query_coverage_end']==length
        pq=[(q['start'],q['start']+q['count']) for q in queries['prefill'] if q['request_id']==rid]
        intervals_cover(pq,0,length)
        dq=[q for q in queries['decode'] if q['request_id']==rid]
        assert dq[0]['start']==length-1 and dq[0]['count']==1
        intervals_cover([(q['start'],q['start']+q['count']) for q in dq],length-1,max(q['start']+q['count'] for q in dq))
        checks.append(dict(request_id=rid,all36_valid_prefix_exact=True,valid_prefix_tokens=length-1,
            complete_prompt_after_D_query=True,hidden_last_recorded=True,
            P_full_prompt_query_coverage=True,Native_last_query_recompute_count=1))
    for role in ('prefill','decode'):assert json.loads((folder/f'KV_lifetime_{role}.json').read_text())['passed']
    write(ROOT/'state/Native_valid_prefix_observer_q22_gate_v2.json',dict(
        passed=True,checks=checks,name=name,evidence='CORRECTNESS_ONLY',
        scope='Unmodified original full36 model. All valid imported prefix bytes match their producer. '
            'The complete prompt KV and last hidden are observed after the original D query. '
            'This does not require BI0 equality between full-query and one-query recomputation paths. '
            'Output quality and serving performance retain their independent gates.'))
    print('NATIVE_VALID_PREFIX_AND_FINAL_STATE_PASS',flush=True)


if __name__=='__main__':main()
