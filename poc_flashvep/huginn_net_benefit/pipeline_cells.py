"""Match Native wire policy and source progress across all candidate methods."""
from .common import *
from .prepare import clone
from .matched_wire_cells import NATIVE_WIRE
from .deferred_v2_candidates import fields
from .optimization_cells import paired


def copy_config(source, name, **options):
    local = ROOT/'configs'/(source+'.json')
    if not local.exists():
        return clone(source, name, **options)
    value = json.loads(local.read_text())
    value.update(name=name, **options)
    return value


def main():
    initialize()
    jobs = []
    for cap in (16, 4):
        for arm in ('zero', 'loop', 'token'):
            options = dict(fields('zero', False), preserve_decode_cohort=True,
                native_transport_limits=NATIVE_WIRE, explicit_graph_teardown=True,
                child_exit_integrity_required=True, skip_dead_partial_coda_prepare=True,
                source_submission_pipeline=True, source_pipeline_minimum_queued_loops=8,
                source_pipeline_interval_ms=5., source_pipeline_control_limit=4,
                cap=cap, workload=f'L2048_O{cap}',
                source_pipeline_scope='Progress only completed, disjoint source transfers behind queued graph work. Native and split arms use the same extension.')
            label = f'pipeline7_output{cap}_{arm}'
            for count in (8, 32):
                source = f'control_huginn_zero_state{count}' if arm == 'zero' else f'matchedwire_{arm}_state{count}'
                cfg = copy_config(source, f'{label}_state{count}', **options)
                cfg.update(request_caps=[cap]*count,
                    qualification_output_prefix=cap < 16,
                    phase='SOURCE_OWNER_PIPELINE_EXACT_STATE_QUALIFICATION')
                if count == 32:
                    cfg['required_qualification'] = f'{label}_state8'
                save(cfg)
                jobs.append(cfg['name'])
            source = 'control_huginn_zero_seed2026110931' if arm == 'zero' else f'selection_matchedwire_{arm}_seed2026110940'
            cfg = paired(copy_config(source, f'{label}_seed2026110940', **options),
                f'{label}_seed2026110940', f'{label}_state32')
            cfg.update(request_caps=[cap]*128,
                phase='SOURCE_OWNER_PIPELINE_SELECTION_NOT_CONFIRMATION')
            if arm != 'zero':
                cfg['completion_deadline_ms'] = 350.
            save(cfg)
            jobs.append(cfg['name'])
    write(ROOT/'state/source_pipeline_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
