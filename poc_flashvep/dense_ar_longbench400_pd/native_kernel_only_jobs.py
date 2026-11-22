"""Test lower-interference GPU activity on the unchanged full400 baseline."""
from .common import ROOT,write
from .experiment import create
from .freeze_capsule import freeze
from .next_jobs import plan


def main():
    capsule,sha=freeze('native_kernel_only_ROI_v3_20261121')
    jobs=[]
    for label in ('Clean1','KernelCopiesROI','Clean2'):
        profile=label=='KernelCopiesROI'
        jobs.append(create(f'Native400_q2.8_{label}_KernelOnly_v3','Native','longbench_balanced400_v1.json',
            pair=(6,7),qps=2.8,seed=2026112261,execution_family='INSTALLED_DEFAULT_COMPILE',
            continuous_source_credit_limit=0,quiescent_witness=True,
            source_capsule=capsule,source_capsule_SHA256=sha,
            kernel_activity=profile,kernel_activity_roles=['prefill','decode'],
            collector_namespace='kernel_only_collector_v3',runtime_api_activity=False,
            activity_copies=True,operator_correlation=False,graph_enclosures=False,
            activity_delay_seconds=30,activity_ROI_seconds=20,
            observer_scope='GPU kernels and copies only. No API or external-correlation tracing. '
                'Require both full-trace and ROI-exposed latency gates within5percent. '
                'Copy activity does not by itself cover every possible network or persistent GPU operation. '
                'An observed gap does not prove an executable decode-safe window.'))
    plan('native_kernel_only_ROI_v3',jobs)
    write(ROOT/'configs/native_kernel_only_observer_v3.json',dict(jobs=jobs,
        full_trace_and_ROI_local_absolute_ratio_limits=[.95,1.05],
        zero_dropped_or_invalid_required=True,first_start_only=True,
        repeat_condition='Run independent triplets only after this collector completes and its point gate passes.',
        evidence='FROZEN_OBSERVER_DESIGN'))
    print('Frozen kernel-only observer triplet.',flush=True)


if __name__=='__main__':main()
