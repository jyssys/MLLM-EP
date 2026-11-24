"""Observe client event-loop stalls without changing token arrival timestamps."""
import asyncio
import gc
import threading
import time

from .common import ROOT, append, write


class ClientHostObserver:
    def __init__(self, name, interval=.005):
        self.name = name
        self.interval = interval
        self.rows = []
        self.collections = {}
        self.task = None

    def on_gc(self, phase, info):
        now = time.monotonic_ns()
        key = (threading.get_ident(), info['generation'])
        if phase == 'start':
            self.collections[key] = now
        else:
            begin = self.collections.pop(key, None)
            if begin is not None:
                self.rows.append(dict(kind='gc', begin_ns=begin, end_ns=now,
                    generation=key[1], thread_id=key[0], collected=info.get('collected'),
                    uncollectable=info.get('uncollectable')))

    async def heartbeat(self):
        while True:
            begin = time.monotonic_ns()
            await asyncio.sleep(self.interval)
            end = time.monotonic_ns()
            self.rows.append(dict(kind='heartbeat', begin_ns=begin, end_ns=end,
                delay_ns=max(0, end-begin-int(self.interval*1e9))))

    async def begin(self):
        gc.callbacks.append(self.on_gc)
        self.task = asyncio.create_task(self.heartbeat())

    def record_append(self, path, result):
        begin = time.monotonic_ns()
        append(path, result)
        self.rows.append(dict(kind='raw_append', begin_ns=begin,
            end_ns=time.monotonic_ns(), request_id=result['request_id'],
            runtime_request_id=result['runtime_request_id']))

    async def close(self):
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        gc.callbacks.remove(self.on_gc)
        target = ROOT/'raw/client_host'/(self.name+'.json')
        assert not target.exists(), 'Do not overwrite a client observation'
        write(target, dict(name=self.name, rows=self.rows, evidence='DIAGNOSTIC',
            interval_seconds=self.interval,
            scope='Client GC, event-loop heartbeat and unchanged raw append enclosures. '
                'No token timestamps are corrected. This observer needs a separate overhead gate.'))
