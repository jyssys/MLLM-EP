"""Keep execution-enclosure diagnostics separate from marker-free selection."""
import argparse
from .common import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sources', nargs='+')
    parser.add_argument('--profile-python', action='store_true')
    parser.add_argument('--decode-period', type=int, default=1)
    args = parser.parse_args()
    assert args.decode_period > 0
    initialize()
    jobs = []
    for source in args.sources:
        cfg = json.loads((ROOT/'configs'/(source+'.json')).read_text())
        assert not cfg.get('state_probe')
        prefix = 'python_cost_' if args.profile_python else 'event_cost_'
        if args.decode_period != 1:
            prefix += f'period{args.decode_period}_'
        cfg.update(name=prefix+source,
            sparse_device_trace=True, sparse_decode_period=args.decode_period, host_cost_trace=True,
            profile_python=args.profile_python, clean_confirmation=False,
            phase='MATCHED_CUDA_EVENT_HOST_COST_DIAGNOSTIC_NOT_MARKER_FREE',
            matched_marker_free_control=source)
        save(cfg)
        jobs.append(cfg['name'])
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
