"""Require full400 coverage and valid imported-state hashes before C3 claims."""
from .experiment import create
from .next_jobs import plan


def main():
    jobs=[]
    for axis in ('Layer','Token','Hybrid'):
        extra=dict(hybrid_suffix_fraction=.125,partition_aligned_chunks=True) if axis=='Hybrid' else {}
        jobs.append(create(f'C3Q6_{axis}4_Natural400_FullState_v1',axis,'longbench_balanced400_v1.json',
            cut=4,pair=(6,7),qps=2.8,seed=2026112201,correctness=True,
            axis_state_capture=True,axis_capture_hidden_last_only=True,
            continuous_source_credit_limit=4,wall_timeout=2400,serving_timeout=2100,**extra))
    plan('continuous_natural400_full_state_qualification_v1',jobs)
    print('Frozen Natural400 full-state C3 qualifications. These are not performance runs.',flush=True)


if __name__=='__main__':main()
