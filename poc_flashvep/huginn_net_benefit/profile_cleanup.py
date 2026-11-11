"""Verify profiler-created target sessions before any signal or cleanup."""
import signal
from .common import *


def sessions(name):
    folder = ROOT / 'raw' / name
    configuration = ROOT / 'configs' / (name+'.json')
    found = set()
    for path in folder.glob('loaded_net_benefit_source_*.json'):
        receipt = json.loads(path.read_text())
        pid = int(path.stem.rsplit('_', 1)[1])
        proc = Path('/proc', str(pid))
        try:
            if proc.stat().st_uid != os.getuid() or receipt['config_sha256'] != digest(configuration):
                continue
            sid = os.getsid(pid)
            leader = Path('/proc', str(sid))
            words = leader.joinpath('cmdline').read_bytes().split(b'\0')
            if (leader.stat().st_uid == os.getuid() and os.getsid(sid) == sid
                    and b'poc_flashvep.looped_pd_headroom_recovery.worker' in words
                    and str(configuration).encode() in words):
                found.add(sid)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return sorted(found)


def dump(name):
    owned = set(sessions(name))
    folder = ROOT / 'raw' / name
    for path in folder.glob('worker_stacks_*.log'):
        pid = int(path.stem.rsplit('_', 1)[1])
        try:
            if Path('/proc', str(pid)).stat().st_uid == os.getuid() and os.getsid(pid) in owned:
                os.kill(pid, signal.SIGUSR1)
        except (FileNotFoundError, ProcessLookupError):
            pass
    write(folder / 'verified_profile_sessions.json', dict(sessions=sorted(owned), time=time.time(),
        proof='Exact worker module, namespace configuration, frozen source receipt, UID, and session leader.'))
    return owned


def cleanup(name):
    owned = dump(name)
    if not owned:
        return
    for sid in owned:
        try:
            os.killpg(sid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    until = time.monotonic()+10
    while sessions(name) and time.monotonic() < until:
        time.sleep(.25)
    for sid in sessions(name):
        os.killpg(sid, signal.SIGKILL)
    write(ROOT / 'raw' / name / 'owned_profile_session_cleanup.json', dict(
        sessions=sorted(owned), time=time.time(),
        scope='Only verified namespace target sessions. No external profiler or user job is selected.'))


def stop_stalled(name):
    cfg = json.loads((ROOT / 'configs' / (name+'.json')).read_text())
    assert cfg.get('local_nsys'), 'This stop applies only to an instrumented profiler trial'
    folder = ROOT / 'raw' / name
    path = folder / 'requests.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    measured = sum(not row.get('warmup') for row in rows)
    age = time.time()-path.stat().st_mtime
    assert measured < cfg['count'] and age > 60, 'The measured cohort is not stalled'
    write(folder / 'manual_profile_watchdog.json', dict(time=time.time(),
        complete_requests=measured, expected=cfg['count'], no_response_progress_seconds=age,
        reason='PROFILE_NIXL_TRANSFER_STALL_AFTER_MEASURED_PROGRESS',
        evidence='Verified source stack is inside nixl_cu12._api.transfer. Destination continues its scheduler loop. NVML sampled 0% utilization on both GPUs. This does not establish physical kernel idle. Root cause below the API remains inconclusive. No complete-run performance or clean physical-idle claim.'))
    cleanup(name)
