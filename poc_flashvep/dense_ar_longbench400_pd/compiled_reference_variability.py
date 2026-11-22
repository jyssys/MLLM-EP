"""Test original compiled-reference repeatability before assigning a transport fault."""
import json
from .common import ROOT, write
from .experiment import create
from .freeze_capsule import freeze
from .run import execute


def main():
    capsule, sha = freeze('compiled_original_reference_variability_v1_20261121')
    jobs = []
    for label, setup in (('Original0', False), ('Setup0', True), ('Original1', False), ('Setup1', True)):
        jobs.append(create(f'Q30_CompiledNative_{label}_N1_KernelPath_v1', 'Native', 'qualification8.json',
            count=1, pair=(0, 1), correctness=True, controlled=True, serial=True,
            output=16, fixed=True, qps=2.8, seed=2026112201,
            execution_family='INSTALLED_DEFAULT_COMPILE',
            d_compute_throttle=True, defer_D_remote_setup=setup,
            state_capture=True, axis_state_capture=False, gpu_state_digest=True,
            numerical_probe=True, numerical_probe_sync=True,
            kernel_activity=True, kernel_activity_roles=['prefill'],
            collector_namespace='kernel_only_collector_v3', activity_copies=False,
            operator_correlation=False, runtime_api_activity=False,
            kernel_activity_seconds=7200, source_capsule=capsule, source_capsule_SHA256=sha,
            wall_timeout=4500, serving_timeout=4000,
            scope='DIAGNOSTIC original compiled Native repeatability. Same prompt,seed,GPU pair and code. '
                'Capture P kernel names/grids and tensor samples before any D remote read. '
                'The P model and scheduler do not install the D-only intervention. '
                'No performance claim. Preserve every strict mismatch and compare original-versus-original first.'))
    write(ROOT / 'configs/compiled_native_reference_variability_v1.json', dict(jobs=jobs,
        evidence='FROZEN_NUMERICAL_DIAGNOSTIC_DESIGN',
        hypotheses=['Original reference numerical path varies across server starts.',
            'The D-only transport intervention corrupts imported state.',
            'A tensor observer or hash schema creates a false mismatch.'],
        falsification='Compare original repeats,P-side tensors before transfer,exact P-to-D valid-prefix hashes, '
            'kernel execution signatures and the changed-policy path separately. Do not overwrite outputs or accept a mismatch without classification.'))
    for name in jobs:
        assert execute(name), name
        print('COMPILED_REFERENCE_DIAGNOSTIC_COMPLETE', name, flush=True)


if __name__ == '__main__':
    main()
