"""Compile official CUDA operators while preserving explicit BF16 boundaries."""
import fcntl,json
from .common import *
from .prepare import save
from .run import execute
from .analyze import compare
from .state_qualification import state_compare
from .eager_feasibility import qualify


def main():
    eager='eager_qwen8_L8000_O16_N8_reference_qual_v16'
    assert (ROOT/'raw'/eager/'dispatcher_result.json').exists()
    jobs=[]
    for count in (8,32):
        for label,mode,cut in [('reference','STATE_REFERENCE',32)]+[(f'c{k}','MIGRATE',k) for k in (35,34,32,28)]:
            c=json.loads((ROOT/'configs'/f'layer_qwen8_L8000_O16_state_reference_c36_N{count}_gpu_v6.json').read_text())
            c.update(name=f'opaque_qwen8_L8000_O16_N{count}_{label}_qual_v17',ar_enforce_eager=False,
                ar_custom_ops=['all'],ar_execution_family='BI_OPAQUE_COMPILED_NO_GRAPH',
                layer_mode=mode,layer_cut=cut,partial_KV_wire=mode=='MIGRATE',separate_suffix_runner=mode=='MIGRATE',
                verify_layer_execution=mode=='MIGRATE',execution_events=True,batch_commands=True,
                state_capture=True,retain_fp32_boundary_residual=False,phase='OPAQUE_OFFICIAL_BF16_OPERATOR_QUALIFICATION')
            jobs.append(save(c))
    write(ROOT/'state/opaque_correctness_plan.json',dict(jobs=jobs,
        scope='Official vLLM CUDA operator configuration applies to Native and candidates. No model kernel is replaced by an approximation. The original aggressive compiled family remains a separate control.'))
    references={};results=[];failed=set()
    try:
        for n in jobs:
            c=json.loads((ROOT/'configs'/(n+'.json')).read_text());count=c['count'];cut=c['layer_cut']
            if c['layer_mode']=='MIGRATE' and count==32 and cut in failed:continue
            if not execute(n,restore=False):break
            if c['layer_mode']=='STATE_REFERENCE':
                references[count]=n
                checks=[compare(eager,n),state_compare(eager,n,'prefill')] if count==8 else [compare(n,references[8],True),state_compare(n,references[8],'prefill',None,True)]
                g=dict(name=n,passed=all(x['passed'] for x in checks),checks=checks,
                    scope='Exact token and bitwise state equivalence to an independent explicit-BF16 full Native reference.')
                write(ROOT/'analysis'/(n+'_opaque_reference_gate.json'),g)
            else:g=qualify(n,references[count])
            results.append(g);write(ROOT/'state/opaque_correctness_progress.json',dict(results=results))
            print('OPAQUE_GATE',n,g['passed'],flush=True)
            if not g['passed']:
                if c['layer_mode']=='STATE_REFERENCE':break
                failed.add(cut)
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
