"""Conservative current-state admission. This is not a safe-window predictor."""
import asyncio
import json
import time


DECODE_CAPS = (1, 2, 4, 8, 16, 32, 64)


class FirstToken:
    def __init__(self):
        self.pending = b''
        self.seen = False

    def feed(self, block):
        if self.seen:
            return False
        self.pending += block
        while b'\n' in self.pending:
            line, self.pending = self.pending.split(b'\n', 1)
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if not data or data == b'[DONE]':
                continue
            value = json.loads(data)
            for choice in value.get('choices', []):
                ids = choice.get('token_ids') or (choice.get('delta') or {}).get('token_ids') or []
                if ids:
                    assert isinstance(ids, list)
                    self.seen = True
                    return True
        return False


class IdleAdmission:
    def __init__(self, batch_one=False, decode_cap=None):
        if batch_one:
            assert decode_cap in (None, 1)
            decode_cap = 1
        assert decode_cap is None or type(decode_cap) is int and decode_cap in DECODE_CAPS
        self.condition = asyncio.Condition()
        self.decode_cap = decode_cap
        self.active = {}
        self.pending = {}
        self.prefill_owner = None
        self.waiting = 0
        self.events = []
        self.failure = None

    @property
    def cap_binding(self):
        return self.decode_cap is not None and len(self.active) >= self.decode_cap

    def state(self):
        return dict(active_http_requests=len(self.active), prefill_owner=self.prefill_owner,
            admission_waiting=self.waiting,failed_closed=self.failure is not None,
            max_active_http_requests=self.decode_cap,cap_binding=self.cap_binding,
            policy='FIFO_P_READY' if self.decode_cap is not None else 'IDLE_SUFFIX',
            scope='Sole-coordinator HTTP lifetime, not engine decode-ready or GPU kernel activity.')

    def can_enter(self, request_id, extra_prefill):
        return (not self.cap_binding and self.prefill_owner is None
                and (not extra_prefill or not self.active)
                and (self.decode_cap is None or next(iter(self.pending)) == request_id))

    def enqueued_blockers(self, extra_prefill):
        return [name for name, blocked in (
            ('CAP_FULL', self.cap_binding),
            ('FIFO_AHEAD', self.decode_cap is not None and bool(self.pending)),
            ('SUFFIX_NEEDS_EMPTY_D', extra_prefill and bool(self.active)),
            ('SUFFIX_PREFILL_IN_PROGRESS', self.prefill_owner is not None),
        ) if blocked]

    async def enter(self, request_id, extra_prefill):
        async with self.condition:
            assert request_id not in self.active and request_id not in self.pending
            begin = time.monotonic()
            enqueued = self.state()
            blocked_on_enqueue = self.enqueued_blockers(extra_prefill)
            self.waiting += 1
            self.pending[request_id] = None
            try:
                await self.condition.wait_for(lambda: self.failure is not None or
                    self.can_enter(request_id, extra_prefill))
                if self.failure is not None:
                    raise RuntimeError('D completion is uncertain. Stop new D admission until this experiment is closed.')
                before = self.state()
                self.active[request_id] = extra_prefill
                if extra_prefill:
                    self.prefill_owner = request_id
                event = dict(request_id=request_id, extra_prefill=extra_prefill, begin=begin,
                    admitted=time.monotonic(), before=before, after=self.state(), enqueued=enqueued,
                    blocked_on_enqueue=blocked_on_enqueue,
                    suffix_isolated=before['active_http_requests']==0 if extra_prefill else None)
                self.events.append(event)
                return event
            finally:
                self.waiting -= 1
                del self.pending[request_id]
                if self.decode_cap is not None:
                    self.condition.notify_all()

    async def first_token(self, request_id):
        async with self.condition:
            assert request_id in self.active
            if self.prefill_owner == request_id:
                self.prefill_owner = None
                self.events.append(dict(request_id=request_id, first_token=time.monotonic(),
                    release='Extra prefill completed. This request remains decode-active.'))
                self.condition.notify_all()

    async def leave(self, request_id,failed=False):
        async with self.condition:
            assert request_id in self.active
            if failed:self.failure=request_id
            del self.active[request_id]
            if self.prefill_owner == request_id:
                self.prefill_owner = None
            self.events.append(dict(request_id=request_id, finished=time.monotonic(), after=self.state()))
            self.condition.notify_all()
