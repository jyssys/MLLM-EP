"""Bind the completed self-test to its immutable compiled collector."""
from .common import *


def main():
    source = ROOT / 'raw/activity_only_collector_selftest/activity_selftest.json'
    result = json.loads(source.read_text())
    build = json.loads((ROOT / 'tools/activity_build.json').read_text())
    assert result['passed']
    assert digest(Path(build['library'])) == build['library_SHA256']
    assert digest(SOURCE / 'kernel_activity.cc') == build['source_SHA256']
    value = dict(selftest=str(source), selftest_SHA256=digest(source), collector_build=build,
                 scope='Completed 200-kernel test of this unchanged collector binary. Model-trace overhead remains untested.')
    target = ROOT / 'tools/activity_validation.json'
    if target.exists():
        assert json.loads(target.read_text()) == value
    else:
        write(target, value)


if __name__ == '__main__':
    main()
