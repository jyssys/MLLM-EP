"""Match event-marker overhead and obtain bounded native kernel traces."""
import copy
import time
from .common import *
from .run import execute
from .static_campaign import configuration


def primary_plans(model):
    return ('native', 'loop3', 'token3072') if model == 'ouro' else ('native', 'loop31', 'token1984')


def setup_collector():
    source = PREVIOUS / 'tools/activity_build.json'
    build = json.loads(source.read_text())
    assert digest(Path(build['library'])) == build['library_SHA256']
    assert digest(PARENT_SOURCE / 'kernel_activity.cc') == build['source_SHA256']
    destination = ROOT / 'tools/activity_build.json'
    if destination.exists():
        assert json.loads(destination.read_text()) == build
    else:
        write(destination, build)
    cfg = configuration('ouro', 'atomic_native', {}, 8, 'activity_only_collector_selftest')
    cfg.update(worker='collector_selftest', activity_selftest=True, collector_test_iterations=100,
               cupti_preallocated_buffers=16, timeout=60, startup_timeout=60,
               phase='COLLECTOR_COVERAGE_SELFTEST_NOT_MODEL_RESULT')
    save(cfg)
    if not execute(cfg['name']):
        return False
    result_path = ROOT / 'raw' / cfg['name'] / 'activity_selftest.json'
    result = json.loads(result_path.read_text())
    assert result['passed']
    write(ROOT / 'tools/activity_validation.json', dict(selftest_SHA256=digest(result_path),
        collector_build=build, scope='New campaign coverage self-test. Model-run overhead still requires a matched control.'))
    return True


def main():
    initialize()
    source = ROOT / 'state/primary_screen.json'
    while time.time() < LONG_START_CUTOFF:
        if source.exists() and json.loads(source.read_text())['status'] == 'PRIMARY_SCREEN_FINISHED':
            break
        time.sleep(5)
    else:
        write(ROOT / 'state/measurement_validation.json', dict(status='NOT_RUN_TIME_BUDGET'))
        return
    primary = json.loads(source.read_text())
    results = dict(status='IN_PROGRESS', markers={}, profiles=[])
    target = ROOT / 'state/measurement_validation.json'
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from ..fluidtoken_loop_split.static_screen import metrics

    for model in ('ouro', 'huginn'):
        results['markers'][model] = {}
        for plan in primary_plans(model):
            selected = next((row for row in primary['results'] if row.get('model') == model
                             and row.get('load') == 'heavy' and row.get('plan') == plan and row.get('passed')), None)
            if selected is None:
                results['markers'][model][plan] = dict(passed=False, status='BLOCKED_PRIMARY_RUN')
                continue
            base = json.loads((ROOT / 'configs' / (selected['name'] + '.json')).read_text())
            pair = []
            for enabled in (True, False):
                cfg = copy.deepcopy(base)
                cfg.update(name=f'j_{model}_{plan}_marker_{"on" if enabled else "off"}128',
                    observe_service_timing=enabled, phase='MATCHED_EVENT_MARKER_OVERHEAD_CONTROL')
                save(cfg)
                row = dict(name=cfg['name'], enabled=enabled, complete=execute(cfg['name']))
                if row['complete']:
                    with quiet_analysis(cfg['name'] + '_marker_metrics'):
                        row['metrics'] = metrics(cfg['name'])
                pair.append(row)
            row = dict(passed=False, controls=pair)
            if all(record['complete'] for record in pair):
                with quiet_analysis(model + '_' + plan + '_marker_parity'):
                    row['output'] = evidence.compare(pair[1]['name'], pair[0]['name'])
                row['wall_change_fraction'] = pair[0]['metrics']['response_span_s'] / pair[1]['metrics']['response_span_s'] - 1
                row['passed'] = row['output']['passed'] and abs(row['wall_change_fraction']) <= .05
                row['scope'] = 'One matched independent pair. Frozen 5% wall screen. This is not a confidence interval or a population overhead bound.'
            results['markers'][model][plan] = row
            write(target, results)

    try:
        collector_ready = setup_collector()
    except (AssertionError, KeyError, FileNotFoundError) as error:
        collector_ready = False
        results['collector_error'] = repr(error)
    results['collector_ready'] = collector_ready
    write(target, results)
    if collector_ready:
        from ..fluidtoken_loop_split.activity_analysis import analyze
        for model in ('ouro', 'huginn'):
            for plan in primary_plans(model)[:2]:
                selected = next((row for row in primary['results'] if row.get('model') == model
                                 and row.get('load') == 'heavy' and row.get('plan') == plan and row.get('passed')), None)
                if selected is None:
                    continue
                cfg = copy.deepcopy(json.loads((ROOT / 'configs' / (selected['name'] + '.json')).read_text()))
                cfg.update(name=f'k_{model}_{plan}_dual_capture128', kernel_activity_seconds=8.,
                    capture_roles=['prefill', 'decode'], capture_arrival_fraction=.35,
                    cupti_preallocated_buffers=16, phase='BOUNDED_DUAL_STAGE_PROFILE_NOT_CLEAN')
                save(cfg)
                row = dict(name=cfg['name'], model=model, plan=plan, complete=execute(cfg['name']), roles={})
                if row['complete']:
                    with quiet_analysis(cfg['name'] + '_CUPTI_parse'):
                        for role in ('prefill', 'decode'):
                            try:
                                row['roles'][role] = analyze(cfg['name'], selected['name'], role)
                            except (AssertionError, KeyError, FileNotFoundError) as error:
                                row['roles'][role] = dict(status='REJECTED_OR_MISSING_ACTIVITY_EVIDENCE',
                                    eligible_for_matched_clean=False, error=repr(error))
                results['profiles'].append(row)
                write(target, results)
    results['status'] = 'MEASUREMENT_VALIDATION_FINISHED'
    write(target, results)


if __name__ == '__main__':
    main()
