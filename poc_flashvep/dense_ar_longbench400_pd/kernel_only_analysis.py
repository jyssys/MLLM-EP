"""Check the separate kernel/copy collector against unchanged observer limits."""
import json
from .common import ROOT,write
from .window_analysis import role_analysis,observer


def main():
    names=[f'Native400_q2.8_{label}_KernelOnly_v3' for label in ('Clean1','KernelCopiesROI','Clean2')]
    if not all((ROOT/'raw'/n/'dispatcher_result.json').exists() for n in names):
        print('PENDING_CLEAN_PROFILE_CLEAN',flush=True);return
    assert all(json.loads((ROOT/'raw'/n/'dispatcher_result.json').read_text())['complete'] for n in names)
    roles=[role_analysis(names[1],r) for r in ('prefill','decode')]
    gate=observer(2.8,roles,names=names)
    for role in roles:role['evidence']='MEASURED_KERNEL_ACTIVITY' if gate['passed'] else 'DIAGNOSTIC'
    value=dict(names=names,roles=roles,observer_gate=gate,decode_safe_executable_window='NOT_IDENTIFIED',
        evidence='MEASURED_KERNEL_ACTIVITY' if gate['passed'] else 'DIAGNOSTIC',
        scope='No CUDA API,external correlation or graph activity tracing. GPU kernels and copies are recorded. '
            'The same5percent full-trace and ROI-exposed-request gates apply. No threshold changes. '
            'One triplet is not a start-level serving confidence interval. No partition state was ready in this Native trace.')
    write(ROOT/'analysis/native_kernel_only_observer_v3_result.json',value)
    print('KERNEL_ONLY_OBSERVER',gate['status'],gate['full_trace_profile_over_mean_clean'],
        gate['matched_exposed_requests_profile_over_mean_clean'],flush=True)
    print('Activity',[(r['role'],r['kernel_busy_fraction'],r['observed_copy_union_ms']) for r in roles],flush=True)


if __name__=='__main__':main()
