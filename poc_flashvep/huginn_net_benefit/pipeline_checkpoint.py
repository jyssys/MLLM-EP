"""Isolate checkpoint cost with the same source progress extension."""
from .common import *


def main():
    initialize()
    jobs = []
    sources = ('matchedwire_checkpoint_state8', 'matchedwire_checkpoint_state32',
        'selection_matchedwire_checkpoint_seed2026110940')
    for source in sources:
        cfg = json.loads((ROOT/'configs'/(source+'.json')).read_text())
        suffix = source.rsplit('_', 1)[1]
        cfg.update(name=f'pipeline7_checkpoint_{suffix}', source_submission_pipeline=True,
            source_pipeline_minimum_queued_loops=8, source_pipeline_interval_ms=5.,
            source_pipeline_control_limit=4, explicit_graph_teardown=True,
            child_exit_integrity_required=True)
        if suffix == 'state32':
            cfg['required_qualification'] = 'pipeline7_checkpoint_state8'
        elif suffix.startswith('seed'):
            cfg['required_qualification'] = 'pipeline7_checkpoint_state32'
        save(cfg)
        jobs.append(cfg['name'])
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
