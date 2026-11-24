"""Conservative current-state admission. This is not a safe-window predictor."""
import asyncio
import json
import time


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
    def __init__(self):
        self.condition = asyncio.Condition()
        self.active = {}
        self.pending = set()
        self.prefill_owner = None
        self.waiting = 0
        self.events = []
        self.failure = None

    def state(self):
        return dict(active_http_requests=len(self.active), prefill_owner=self.prefill_owner,
            admission_waiting=self.waiting,failed_closed=self.failure is not None,
            scope='Sole-coordinator HTTP lifetime, not engine decode-ready or GPU kernel activity.')

    async def enter(self, request_id, extra_prefill):
        async with self.condition:
            assert request_id not in self.active and request_id not in self.pending
            begin = time.monotonic()
            self.waiting += 1
            self.pending.add(request_id)
            try:
                await self.condition.wait_for(lambda: self.failure is not None or
                    self.prefill_owner is None and (not extra_prefill or not self.active))
                if self.failure is not None:
                    raise RuntimeError('D completion is uncertain. Stop new D admission until this experiment is closed.')
                before = self.state()
                self.active[request_id] = extra_prefill
                if extra_prefill:
                    self.prefill_owner = request_id
                event = dict(request_id=request_id, extra_prefill=extra_prefill, begin=begin,
                    admitted=time.monotonic(), before=before, after=self.state())
                self.events.append(event)
                return event
            finally:
                self.waiting -= 1
                self.pending.remove(request_id)

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
