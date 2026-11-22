"""Keep continuous decode admission in the stronger whole-prefill control."""
import json
import time
from .common import ROOT,write,dispatch_allowed
from .experiment import create
from .freeze_capsule import freeze
from .run import execute
from .quality_gate import compare as quality
from .compute_throttle_qualification import no_mixed


def main():
    capsule,sha=freeze('whole_deflection_continuous_qualification_v2_20261121')
    old=ROOT/'state/whole_prefill_deflection_gate_v1.json'
    while not (old.exists() and json.loads(old.read_text()).get('passed')):
        assert dispatch_allowed();time.sleep(5)
    results=[]
    for n,dataset in ((32,'continuous_qualification32_natural.json'),(400,'longbench_balanced400_v1.json')):
        name=create(f'Q25_WholeDeflection_Continuous_Natural{n}_v2','Native',dataset,
            pair=(4,5),qps=2.8,seed=2026112201,correctness=True,
            execution_family='INSTALLED_DEFAULT_COMPILE',state_capture=True,axis_state_capture=False,
            gpu_state_digest=True,whole_prefill_deflection=True,d_compute_throttle=True,
            continuous_source_credit_limit=4,source_capsule=capsule,source_capsule_SHA256=sha,
            wall_timeout=4500,serving_timeout=4000)
        assert execute(name),name
        folder=ROOT/'raw'/name
        data=json.loads((ROOT/'data'/dataset).read_text());records=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
        assert len(records)==n
        direct=[json.loads(s) for s in (folder/'whole_prefill_deflections.jsonl').read_text().splitlines()]
        assert direct,'The control must execute real D-local prefill'
        direct_ids={r['request_id'] for r in direct}
        source=json.loads((folder/'native_valid_prefix_prefill.json').read_text())['requests']
        dest=json.loads((folder/'native_valid_prefix_decode.json').read_text())['requests']
        full=json.loads((folder/'prompt_state_decode.json').read_text())['requests']
        for i,row in enumerate(data):
            rid='q'+str(i);assert len(full[rid]['KV'])==36 and full[rid]['query_coverage_end']==row['input_tokens']
            if rid not in direct_ids:assert source[rid]['KV']==dest[rid]['KV'],(rid,'Native imported prefix mismatch')
        for role in ('prefill','decode'):assert json.loads((folder/f'KV_lifetime_{role}.json').read_text())['passed']
        mix=no_mixed(name);score=quality(name);assert score['passed'],score
        results.append(dict(name=name,n=n,deflections=len(direct),Native_handoffs=n-len(direct),
            all36_full_valid_prompt_states=True,Native_valid_prefix_exact=True,compute_deferral=mix,quality=score))
        write(ROOT/'state/whole_prefill_deflection_continuous_gate_v2.json',dict(time=time.time(),
            passed=len(results)==2,full400_passed=len(results)==2,results=results,
            inherited_N1_N8_strict_proof=json.loads(old.read_text())['results'][:2],
            evidence='CORRECTNESS_ONLY',
            scope='The D-local full36 numerical path retains its independent strict N1/N8 proof. '
                'The new policy admits normal Native one-query continuations and decode continuously. '
                'Large D-local prefill chunks use the installed compute-throttle path. '
                'This is stronger than whole-request serial D admission. All native imports exclude the uncomputed last position before comparison. '
                'No speed or client-ITL success follows from correctness.'))
    print('CONTINUOUS_WHOLE_PREFILL_DEFLECTION_QUALIFIED',flush=True)


if __name__=='__main__':main()
