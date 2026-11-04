"""Reuse the qualified runner and its bounded owned-process cleanup."""
import argparse
import os
import zipfile
import threading
import time
from .common import *


def execute(name, keep_quiet=True):
    initialize()
    from ..looped_pd_autonomous_headroom import run as inherited
    inherited.ROOT = ROOT
    inherited.GPU_DEADLINE = GPU_DEADLINE
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    if (time.time() >= LONG_START_CUTOFF or time.time() >= GPU_DEADLINE - 180 or
            time.time() + cfg.get('timeout', 300) + cfg.get('startup_timeout', 240) >= GPU_DEADLINE):
        folder = ROOT / 'raw' / name
        folder.mkdir(parents=True, exist_ok=True)
        receipt = folder / 'dispatcher_result.json'
        if not receipt.exists():
            write(receipt, dict(complete=False, status='NOT_RUN_TIME_BUDGET',
                reason='The new-job cutoff passed, or the configured timeout and startup allowance cannot fit before the GPU stop boundary.',
                time=time.time()))
        return False
    original = inherited.environment
    def environment(gpus):
        env = original(gpus)
        env['FLUIDTOKEN_LOOP_WORKER'] = '1'
        if cfg.get('native_stack_diagnostic'):
            env['FLUIDTOKEN_DEBUGGER_OWNER_PID'] = str(os.getpid())
        env['PYTHONPATH'] = str(SOURCE / 'bootstrap') + os.pathsep + env['PYTHONPATH']
        folder = ROOT / 'raw' / name
        with zipfile.ZipFile(folder / 'new_campaign_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(list(SOURCE.rglob('*.py')) + list(SOURCE.rglob('*.cc'))):
                archive.write(path, str(path.relative_to(REPO)))
        write(folder / 'new_campaign_source_hashes.json',
              {str(path): digest(path) for path in sorted(list(SOURCE.rglob('*.py')) + list(SOURCE.rglob('*.cc')))})
        return env
    inherited.environment = environment
    stop = threading.Event()
    def inspect_stall():
        folder = ROOT / 'raw' / name
        signature, changed, recorded = None, time.monotonic(), set()
        while not stop.wait(2):
            path = folder / 'requests.jsonl'
            size = path.stat().st_size if path.exists() else None
            if size != signature:
                signature, changed = size, time.monotonic()
            if size is None or time.monotonic() - changed < 30:
                continue
            if cfg.get('kernel_activity_seconds') and time.monotonic() - changed > 60:
                from ..looped_pd_headroom_recovery.run import measured_requests_complete
                if not measured_requests_complete(folder, cfg['count']):
                    import signal
                    active = json.loads((ROOT / 'state/active.json').read_text())
                    pid = active['pid']
                    proc = Path('/proc', str(pid))
                    if (active['name'] == name and proc.stat().st_uid == os.getuid()
                            and os.getsid(pid) == pid and os.getpgid(pid) == pid):
                        write(folder / 'activity_watchdog.json', dict(
                            reason='ACTIVITY_CAPTURE_NO_RESPONSE_PROGRESS_60S', pid=pid, time=time.time()))
                        os.killpg(pid, signal.SIGTERM)
                        return
            for record in folder.glob('stage_owner_*.json'):
                value = json.loads(record.read_text())
                pid = value['pid']
                if pid in recorded:
                    continue
                try:
                    proc = Path('/proc', str(pid))
                    words = (proc / 'cmdline').read_bytes().decode().split('\0')
                    if (proc.stat().st_uid != os.getuid()
                            or str(value['payload']) not in words
                            or Path(value['payload']).parent != folder):
                        continue
                    recorded.add(pid)
                    result = subprocess.run([
                        '/home/esjung/.local/opt/fluidtoken-debug-tools/bin/py-spy',
                        'dump', '--native', '--pid', str(pid)],
                        capture_output=True, text=True, timeout=15)
                    write(folder / f'native_stall_stack_{pid}.json', dict(
                        returncode=result.returncode, stdout=result.stdout,
                        stderr=result.stderr, time=time.time(),
                        scope='Diagnostic attachment after 30 seconds without response progress. Not clean timing.'))
                    debugger = Path('/home/esjung/.local/opt/fluidtoken-native-debug/root/usr/bin/gdb')
                    if debugger.exists():
                        diagnostic_env = dict(os.environ,
                            LD_LIBRARY_PATH='/home/esjung/.local/opt/fluidtoken-native-debug/root/usr/lib/x86_64-linux-gnu',
                            DEBUGINFOD_URLS='')
                        result = subprocess.run([str(debugger), '-nx', '-batch',
                            '-ex', 'set auto-load safe-path /nonexistent',
                            '-ex', 'set pagination off', '-ex', 'set print frame-arguments none',
                            '-ex', 'thread apply all bt 24', '-ex', 'detach', '-p', str(pid)],
                            env=diagnostic_env, capture_output=True, text=True, timeout=10)
                        write(folder / f'native_all_threads_{pid}.json', dict(
                            returncode=result.returncode, stdout=result.stdout,
                            stderr=result.stderr, time=time.time(),
                            scope='Bounded native all-thread debugger attachment. No local variables are requested. Not clean timing.'))
                except (OSError, subprocess.TimeoutExpired) as error:
                    write(folder / f'native_stall_stack_error_{pid}.json', dict(error=repr(error)))
    monitor = None
    if cfg.get('native_stack_diagnostic') or cfg.get('kernel_activity_seconds'):
        monitor = threading.Thread(target=inspect_stall, daemon=True)
        monitor.start()
    original_launch = inherited.subprocess.Popen
    if cfg.get('activity_selftest'):
        def launch(command, *args, **kwargs):
            if isinstance(command, list) and 'poc_flashvep.looped_pd_autonomous_headroom.collector_selftest' in command:
                command = list(command)
                command[command.index('poc_flashvep.looped_pd_autonomous_headroom.collector_selftest')] = 'poc_flashvep.fluidtoken_loop_split.activity_selftest'
            return original_launch(command, *args, **kwargs)
        inherited.subprocess.Popen = launch
    try:
        return inherited.execute(name, keep_quiet)
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=20)
        inherited.environment = original
        inherited.subprocess.Popen = original_launch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    raise SystemExit(0 if execute(args.name, not args.restore) else 1)
