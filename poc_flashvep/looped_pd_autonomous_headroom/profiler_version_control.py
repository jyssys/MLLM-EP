"""Retry one failed capture with the official CUDA 12.9 bundled collector."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .resume_joint_review import analyze_capture
from .verify_long_probes import verify
from .independent_joint_repeats import captures

TOOL = Path('/home/esjung/.local/opt/looped_pd_profiler_control/nsight_systems-linux-x86_64-2025.1.3.140-archive/nsight-systems/2025.1.3/bin/nsys')


def main():
    control = ROOT/'state/batch_capacity_control.json'
    while time.time() < LONG_START_CUTOFF:
        if control.exists() and json.loads(control.read_text()).get('stage') in (
                'COMPLETE_REVIEW_REQUIRED', 'FAILED_REVIEW_REQUIRED'):
            break
        time.sleep(5)
    assert TOOL.exists()
    version = subprocess.check_output([str(TOOL), '--version'], text=True).strip()
    assert '2025.1.3' in version
    os.environ['PATH'] = str(TOOL.parent)+os.pathsep+os.environ['PATH']
    activate()
    from ..looped_pd_atomic_prefill import evidence
    original = json.loads((ROOT/'configs/short_huginn_s1_nsys384.json').read_text())
    template = json.loads((ROOT/'configs/short_huginn_s1_baseline384.json').read_text())
    ledger = []
    for start, pair in enumerate(((4, 5), (6, 7), (4, 5)), 1):
        if time.time() >= LONG_START_CUTOFF:
            break
        if start == 1:
            baseline = 'short_huginn_s1_baseline384'
            clean = 'short_huginn_s1_probe384'
            cfg = copy.deepcopy(original)
        else:
            base = copy.deepcopy(template)
            base.update(name=f'nsys2025_short_huginn_s{start}_baseline384',
                gpus=list(pair), arrival_seed=2026111500+start, independent_start=start)
            baseline = put(base, 384, base['arrival_rate'], 'short_confirmation')
            write(ROOT/'state/profiler_version_control.json', dict(stage='BASELINE', name=baseline, time=time.time()))
            assert execute(baseline)
            probe = copy.deepcopy(base)
            probe.update(name=f'nsys2025_short_huginn_s{start}_probe384',
                D_probe_tokens=16, D_probe_count=12, D_probe_interval_s=4.)
            clean = put(probe, 384, base['arrival_rate'], 'short_confirmation')
            assert execute(clean) and evidence.compare(baseline, clean)['passed']
            checked = verify(clean)
            assert checked.get('exact') and checked.get('full_work')
            cfg = copy.deepcopy(original)
            cfg.update(gpus=list(pair), arrival_seed=2026111500+start, independent_start=start)
        cfg.update(name=f'nsys2025_short_huginn_s{start}_nsys384', matched_clean=clean,
            profiler_executable=str(TOOL), profiler_version=version,
            profiler_archive_SHA256='dded4227619340307be0ba5bc4e23bcbc966e2df3763170ebb20410c2b54754e')
        name = put(cfg, 384, cfg['arrival_rate'], 'short_confirmation')
        write(ROOT/'state/profiler_version_control.json', dict(stage='PROFILE', name=name, time=time.time()))
        row = dict(name=name, model='huginn', baseline=baseline, clean=clean, status='PROFILE_FAILED', profiler=version)
        if execute(name):
            row.update(analyze_capture(cfg, clean))
            row['measured_captures'] = {n: captures(n) for n in (baseline, clean, name)}
            row['accepted_joint_screen'] = bool(row['joint_screen_pass'] and not any(row['measured_captures'].values()))
            row['status'] = 'COMPLETE_REVIEW_REQUIRED'
        ledger.append(row)
        write(ROOT/'state/profiler_version_control_results.json', ledger)
        if not row.get('accepted_joint_screen'):
            break
    write(ROOT/'state/profiler_version_control.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/profiler_version_control.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
