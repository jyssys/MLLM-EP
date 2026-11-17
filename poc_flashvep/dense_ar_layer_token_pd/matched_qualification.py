"""Qualify the adjacent aligned cuts required by exact GEMM-row controls."""
import fcntl
from .common import QUIET_LOCK, resources
from .qualification import configuration, gate, job


def main():
    try:
        for count in (8, 32):
            for lower, upper in ((1776, 1792), (2656, 2672)):
                name = f'Token_plan_{lower}_{upper}_state_N{count}_v1'
                plan = [lower if (i // 4) % 2 == 0 else upper for i in range(count)]
                cfg = configuration(name, count=count, suffix=lower)
                cfg['token_suffix_by_request'] = plan
                assert job(cfg)
                assert gate(name, 'Native_state_N8_O16_v1' if count == 8 else 'Native_state_N32_O16_v1')
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
