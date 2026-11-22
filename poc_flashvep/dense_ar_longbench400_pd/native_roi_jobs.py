"""Check clean/profile/clean interference on full400 continuous Native traces."""
from .experiment import create
from .freeze_capsule import freeze
from .next_jobs import plan


def main():
    capsule,sha=freeze('native_continuous_ROI_v1_20261121')
    for qps,pair in ((1.8,(2,3)),(2.8,(6,7))):
        jobs=[]
        for label in ('Clean1','BothGPU_ROI','Clean2'):
            extra={}
            if label=='BothGPU_ROI':
                extra=dict(kernel_activity=True,kernel_activity_roles=['prefill','decode'],
                    runtime_api_activity=True,activity_copies=False,graph_enclosures=False,
                    activity_delay_seconds=30,activity_ROI_seconds=20)
            jobs.append(create(f'Native400_q{qps}_{label}_ROI_v1','Native','longbench_balanced400_v1.json',
                pair=pair,qps=qps,seed=2026112261,execution_family='INSTALLED_DEFAULT_COMPILE',
                continuous_source_credit_limit=0,quiescent_witness=True,
                source_capsule=capsule,source_capsule_SHA256=sha,
                clean_profile_clean_contract='Same full400 arrival and generation contract. '
                    'Require every primary latency ratio to stay within5percent of the clean mean. '
                    'Also inspect requests and ITLs that intersect the actual ROI. '
                    'A full-trace average must not hide local observer interference. '
                    'Kernel inactivity is not a prediction of a decode-safe window.',**extra))
        plan(f'native400_q{qps}_ROI_v1',jobs)
    print('Frozen two full400 continuous Native ROI triplets.',flush=True)


if __name__=='__main__':main()
