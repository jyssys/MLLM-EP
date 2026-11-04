"""Record owned process wait channels without attaching a debugger."""
import argparse
import time
from .common import *


def snapshot(name):
    folder = ROOT / 'raw' / name
    owner = json.loads((folder / 'stage_owner_prefill.json').read_text())
    process = Path('/proc', str(owner['pid']))
    assert process.stat().st_uid == os.getuid()
    assert owner['payload'] in (process / 'cmdline').read_bytes().decode().split('\0')
    values = []
    for task in (process / 'task').iterdir():
        try:
            values.append(dict(tid=int(task.name), name=(task / 'comm').read_text().strip(),
                               wchan=(task / 'wchan').read_text().strip()))
        except FileNotFoundError:
            continue
    write(folder / f'wait_channels_{time.time_ns()}.json', dict(pid=owner['pid'], time=time.time(), tasks=values,
        scope='Linux wait channels only. A futex wait does not identify the lock owner or prove a deadlock cycle.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    snapshot(parser.parse_args().name)
