"""Isolate the Huginn scheduler correction before changing graph selection."""
import subprocess
from .common import *
from .prepare import configuration
from .qualification import qualify


def guard():
    path = ROOT / 'state/deadline_guard_owner.json'
    if path.exists():
        record = json.loads(path.read_text())
        if Path('/proc', str(record['pid'])).exists():
            return
    with (ROOT / 'logs/deadline_guard.log').open('a') as log:
        child = subprocess.Popen(['/home/esjung/.venvs/looped-pd-abc/bin/python', '-m',
            'poc_flashvep.native_split_deep_validation.deadline_guard'], cwd=REPO,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(path, dict(pid=child.pid, deadline=GPU_DEADLINE, time=time.time()))


def main():
    initialize()
    guard()
    results = {}
    for method in ('native', 'token1984'):
        cfg = configuration('huginn', method, 8, f'a_huginn_{method}_scheduler_recovery8',
                            state=True, policy=None, simultaneous=True)
        save(cfg)
        reference = 'a_huginn_native_scheduler_recovery8' if method != 'native' else None
        results[method] = qualify(cfg['name'], reference)
        write(ROOT / 'state/isolated_scheduler_recovery.json', results)
        if not results[method]['passed']:
            break
    from ..looped_pd_headroom_recovery.common import restore_if_idle
    restore_if_idle()
    print(json.dumps({key: value['passed'] for key, value in results.items()}), flush=True)


if __name__ == '__main__':
    main()
