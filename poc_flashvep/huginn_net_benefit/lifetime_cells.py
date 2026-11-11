"""Create new namespaces for explicit graph teardown and checked worker exits."""
import argparse
from .common import *


def prepare(names):
    jobs = []
    for source in names:
        cfg = json.loads((ROOT/'configs'/(source+'.json')).read_text())
        cfg.update(name='lifetime3_'+source, explicit_graph_teardown=True,
            child_exit_integrity_required=True, lifetime_parent_config=source)
        required = cfg.get('required_qualification')
        if required and required in names:
            cfg['required_qualification'] = 'lifetime3_'+required
        save(cfg)
        jobs.append(cfg['name'])
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    args = parser.parse_args()
    initialize()
    jobs = prepare(args.names)
    write(ROOT/'state'/f'lifetime3_jobs_{time.time_ns()}.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
