"""Record the actual CUPTI libraries mapped by an owned process."""
import argparse
from .common import *


def record(name, pid):
    proc = Path('/proc', str(pid))
    assert proc.stat().st_uid == os.getuid()
    paths = sorted({line.split()[-1] for line in (proc / 'maps').read_text().splitlines()
                    if '/libcupti.so' in line})
    values = [dict(path=path, SHA256=digest(Path(path))) for path in paths if Path(path).is_file()]
    target = ROOT / 'raw' / name / f'collector_library_mapping_{pid}.json'
    result = dict(pid=pid, mapped_CUPTI_libraries=values,
                  scope='Actual process mappings. Collector and library changes are not isolated causal effects.')
    if target.exists():
        assert json.loads(target.read_text()) == result
    else:
        write(target, result)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('pid', type=int)
    args = p.parse_args()
    record(args.name, args.pid)
