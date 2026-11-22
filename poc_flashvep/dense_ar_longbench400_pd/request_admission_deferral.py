"""Delay new request registration, not token delivery, during active decode."""
import atexit
from collections import deque
import time
from .common import ROOT, write


class AdmissionConnection:
    """Keep received additions until no output-started request remains running."""
    def __init__(self, connection, decode_active):
        self.connection = connection
        self.decode_active = decode_active
        self.pending = deque()
        self.ready = deque()
        self.rows = []

    def poll(self, timeout=0.0):
        active = self.decode_active()
        if self.pending and not active:
            command = self.pending.popleft()
            self.ready.append(command)
            self.rows.append(dict(event='RELEASE_COMMAND', kind=command['kind'],
                request_id=command.get('request_id'), time=time.monotonic(),
                pending=len(self.pending)))
        if self.ready:
            return True
        while self.connection.poll(timeout):
            timeout = 0.0
            begin = time.monotonic()
            command = self.connection.recv()
            end = time.monotonic()
            kind = command['kind']
            # Stop must follow previously received additions. Do not discard a
            # deferred request or prevent normal engine/connector retirement.
            hold = (kind == 'add' and active) or (kind == 'stop' and bool(self.pending))
            self.rows.append(dict(event='RECEIVE_COMMAND', kind=kind,
                request_id=command.get('request_id'), time=end,
                receive_host_ms=1000 * (end - begin), decode_active=active,
                deferred=hold, input_tokens=len(command.get('input_ids', ()))))
            if hold:
                self.pending.append(command)
                continue
            self.ready.append(command)
            return True
        return False

    def recv(self):
        assert self.ready, 'Call poll before receiving a deferred worker command'
        return self.ready.popleft()

    def send(self, value):
        return self.connection.send(value)

    def close(self):
        return self.connection.close()


def wrap(connection, cfg, role):
    if role != 'decode' or not cfg.get('defer_D_request_registration'):
        return connection
    assert cfg.get('d_compute_throttle') and cfg.get('defer_D_remote_setup')
    assert not cfg.get('d_quiescent_admission')
    from vllm.v1.core.sched.scheduler import Scheduler
    original = Scheduler.schedule
    owner = [None]

    def scheduled(scheduler, *args, **kwargs):
        owner[0] = scheduler
        return original(scheduler, *args, **kwargs)

    def active():
        scheduler = owner[0]
        return scheduler is not None and any(len(r.output_token_ids) > 0
            for r in scheduler.running)

    Scheduler.schedule = scheduled
    result = AdmissionConnection(connection, active)

    def finish():
        passed = not result.pending and not result.ready
        write(ROOT / 'raw' / cfg['name'] / 'D_request_registration_deferral.json',
            dict(rows=result.rows, pending=len(result.pending), ready=len(result.ready),
                 passed=passed, evidence='HOST_ADMISSION_POLICY_AUDIT',
                 scope='Delay engine.add_request and its input processing for new additions while '
                       'the scheduler retains an output-started running request. IPC receive still runs. '
                       'Do not alter model inputs,outputs,client timestamps,KV metadata or ACK handling. '
                       'Decode engine steps and connector retirement continue. This is not a safe-window predictor. '
                       'All received additions drain before a stop command. Cancellation/failure safety needs separate tests.'))
        assert passed, 'Do not discard deferred request registration'

    atexit.register(finish)
    return result


def test():
    class FakeConnection:
        def __init__(self, commands):
            self.commands = deque(commands)
            self.sent = []
        def poll(self, timeout=0.0):
            return bool(self.commands)
        def recv(self):
            return self.commands.popleft()
        def send(self, value):
            self.sent.append(value)
        def close(self):
            pass
    state = [True]
    first = dict(kind='add', request_id='q1', input_ids=[3, 5])
    second = dict(kind='add', request_id='q2', input_ids=[7, 9])
    cleanup = dict(kind='cleanup_remote', request_id='retired')
    stop = dict(kind='stop')
    raw = FakeConnection([first, second, cleanup, stop])
    proxy = AdmissionConnection(raw, lambda: state[0])
    assert proxy.poll() and proxy.recv() is cleanup
    assert not proxy.poll() and list(proxy.pending) == [first, second, stop]
    token = dict(kind='output', count=1, time=7.5, new_ids=[8])
    proxy.send(token)
    assert raw.sent == [token] and raw.sent[0] is token
    state[0] = False
    for expected in (first, second, stop):
        assert proxy.poll() and proxy.recv() is expected
    assert not proxy.poll() and not proxy.pending and not proxy.ready
    idle = AdmissionConnection(FakeConnection([first, second]), lambda: False)
    assert idle.poll() and idle.recv() is first
    assert idle.poll() and idle.recv() is second
    print('REQUEST_REGISTRATION_FIFO_CONTROL_AND_OUTPUT_PASS', flush=True)


if __name__ == '__main__':
    test()
