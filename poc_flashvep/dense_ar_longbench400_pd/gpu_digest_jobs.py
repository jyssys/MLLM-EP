"""Verify GPU leaf digests against every CPU byte before full400 use."""
from .experiment import create
from .next_jobs import plan


def main():
    common=dict(pair=(6,7),output=16,fixed=True,serial=True,correctness=True,
        gpu_state_digest=True,verify_GPU_digest_with_CPU=True,gpu_digest_selftest=True,
        wall_timeout=1800,serving_timeout=1500)
    jobs=[create('Q9_GPUdigest_Full36_BoundaryTeacher_N8_v1','Native','qualification8.json',
        reference_single=True,token_teacher_cut=4,teacher_suffix_fraction=.125,
        state_capture=True,axis_state_capture=False,numerical_probe=True,
        numerical_probe_sync=True,kv_layout='HND',**common),
        create('Q9_GPUdigest_Hybrid4_N8_PostConsumptionParity_v1','Hybrid','qualification8.json',cut=4,
            controlled=True,hybrid_suffix_fraction=.125,partition_aligned_chunks=True,
            axis_state_capture=True,axis_capture_hidden_last_only=True,axis_import_hash_after_model=True,**common)]
    plan('gpu_digest_full_state_validation_v1',jobs)
    print('Frozen GPU-digest CPU verification and post-consumption state proof.',flush=True)


if __name__=='__main__':main()
