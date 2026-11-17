"""Run heavy CPU state audits outside every GPU timing lease."""
import fcntl
from .common import QUIET_LOCK, ROOT, resources, write


def main():
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        owner = resources()
        if not owner.processes():
            owner.restore(list(range(8)))
        from .qualification import gate
        values = []
        for suffix in (512, 1024, 2048, 4096):
            name = f'Token{suffix}_state_N32_O16_v1'
            values.append(dict(name=name, passed=gate(name, 'Native_state_N32_O16_v1')))
        for suffix in (2048, 4096):
            name = f'shared_Token{suffix}_state_N32_O64_v1'
            values.append(dict(name=name, passed=gate(name, 'Native_state_N32_O16_v1')))
        from .tensor_analysis import analyze
        tp = analyze('TP1_first_tensor_witness_req4_v5', 'TP2_first_tensor_witness_req4_v4')
        write(ROOT / 'analysis/strict_byte_state_audit.json', dict(qualified=values,
            all_passed=all(v['passed'] for v in values), TP_first_byte_differences=tp['first_differences'],
            scope='CPU audit holds the quiet-lane lease. No serving latency is collected. Byte checks distinguish signed zero. Earlier raw results remain unchanged.'))
        assert all(v['passed'] for v in values)


if __name__ == '__main__':
    main()
