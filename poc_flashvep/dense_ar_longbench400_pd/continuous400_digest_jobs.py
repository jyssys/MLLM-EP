"""Repeat full400 state qualification with a CPU-verified GPU digest observer."""
import json
from .common import ROOT,write
from .experiment import create
from .next_jobs import plan


def main():
    assert json.loads((ROOT/'state/GPU_digest_q9_gate.json').read_text())['passed']
    jobs=[]
    for axis in ('Layer','Token','Hybrid'):
        extra=dict(hybrid_suffix_fraction=.125,partition_aligned_chunks=True) if axis=='Hybrid' else {}
        jobs.append(create(f'C3Q6_{axis}4_Natural400_GPUdigest_v2',axis,'longbench_balanced400_v1.json',
            cut=4,pair=(4,5),qps=2.8,seed=2026112201,correctness=True,
            axis_state_capture=True,axis_capture_hidden_last_only=True,
            gpu_state_digest=True,verify_GPU_digest_with_CPU=False,axis_import_hash_after_model=True,
            continuous_source_credit_limit=4,wall_timeout=4500,serving_timeout=4000,
            observer_scope='Correctness only. Every valid KV byte enters the SHA256-leaf digest. '
                'Q9 compares each actual model-tensor leaf digest with CPU hashlib. '
                'This is not the raw-byte-stream SHA256 and is not a timing measurement.',**extra))
    plan('continuous_natural400_GPUdigest_qualification_v2',jobs)
    write(ROOT/'configs/continuous400_observer_revision_v2.json',dict(
        previous='C3Q6_Layer4_Natural400_FullState_v1',previous_status='FAILED',
        previous_completed_requests=237,previous_timeout_seconds=2100,
        policy='Preserve the previous timeout and all partial logs. Use a different,validated read-only observer. '
            'Do not reinterpret previous237 requests as a full400 pass.',
        jobs=jobs,evidence='FROZEN_CORRECTNESS_DESIGN'))
    print('Frozen full400 GPU-digest correctness runs.',flush=True)


if __name__=='__main__':main()
