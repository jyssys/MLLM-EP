"""Qualify the exact common Native compute-only policy before static selection."""
import json
from .common import ROOT,write
from .experiment import create
from .freeze_capsule import freeze
from .run import execute
from .remote_setup_qualification import native_state
from .compute_throttle_qualification import no_mixed
from .quality_gate import compare as quality
from .metrics import summarize


def main():
    capsule,sha=freeze('native_compute_only_common_control_v1_20261121')
    common=dict(pair=(4,5),execution_family='INSTALLED_DEFAULT_COMPILE',
        continuous_source_credit_limit=0,d_compute_throttle=True,defer_D_remote_setup=False,
        source_capsule=capsule,source_capsule_SHA256=sha)
    proofs=[]
    path=ROOT/'state/Native_compute_only_common_policy_gate_v1.json'
    for n,dataset in ((32,'continuous_qualification32_natural.json'),(400,'longbench_balanced400_v1.json')):
        name=create(f'Q35_NativeComputeOnly_Natural{n}_v1','Native',dataset,qps=2.8,seed=2026112201,
            correctness=True,state_capture=True,axis_state_capture=False,gpu_state_digest=True,**common)
        assert execute(name),name
        state=native_state(name)
        score=quality(name)
        assert state['passed'] and score['passed']
        proofs.append(dict(name=name,state=state,quality=score,compute_rule=no_mixed(name)))
        write(path,dict(passed=len(proofs)==2,full400_passed=len(proofs)==2,proofs=proofs,
            evidence='CORRECTNESS_ONLY',scope='Original Native model,no READ deferral,no D running cap. '
                'The large-prefill rule still exempts the normal one-query Native prompt recomputation. '
                'Normal completion and task quality only. No performance CI.'))
    for qps in (1.8,2.8):
        name=create(f'NativeCompute_Discovery200_q{qps}_v1','Native','balanced400_discovery.json',
            qps=qps,seed=2026112291,discovery_only=True,**common)
        assert execute(name),name
        m=summarize(name)
        print('NATIVE_COMPUTE_CONTROL',qps,m['TTFT_p95_ms'],m['ITL_p99_ms'],m['TPOT_p95_ms'],
            m['SLO_qualified_output_tokens_per_second'],quality(name)['passed'],flush=True)
    print('NATIVE_COMPUTE_COMMON_COMPLETE',flush=True)


if __name__=='__main__':
    main()
