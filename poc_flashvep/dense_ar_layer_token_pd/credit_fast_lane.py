"""Handle host-only credit RPCs while the main worker waits for GPU output.

Keep the existing allocator and credit equations. Serialize allocator and
ledger operations. The control thread must not call CUDA or NIXL.
"""
import atexit
import json
import queue
import threading
import time
from .common import ROOT, write


class ControlChannel:
    def __init__(self, channel, handler):
        self.channel, self.handler = channel, handler
        self.pending = queue.Queue()
        self.send_lock = threading.RLock()
        self.error = None
        self.thread = threading.Thread(target=self.read, daemon=True, name='D_credit_host_only')
        self.thread.start()

    def read(self):
        try:
            while True:
                command = self.channel.recv()
                if command['kind'] in ('reserve', 'commit', 'cancel_reservation'):
                    self.send(self.handler(command))
                else:
                    self.pending.put(command)
                    if command['kind'] == 'stop':
                        break
        except (EOFError, OSError) as error:
            self.error = error
            self.pending.put(None)
        except BaseException as error:
            self.error = error
            self.pending.put(None)

    def poll(self, timeout=0.):
        if self.error:
            raise RuntimeError('Credit control thread failed') from self.error
        if not self.pending.empty():
            return True
        if timeout:
            until = time.monotonic() + timeout
            while time.monotonic() < until and self.pending.empty() and self.error is None:
                time.sleep(.0002)
        return not self.pending.empty()

    def recv(self):
        value = self.pending.get()
        if value is None:
            raise RuntimeError('Credit control channel closed') from self.error
        return value

    def send(self, value):
        with self.send_lock:
            return self.channel.send(value)

    def close(self):
        self.channel.close()
        self.thread.join(timeout=1.)


def locked(lock, method):
    def call(*args, **kwargs):
        with lock:
            return method(*args, **kwargs)
    return call


def install(cfg, role):
    if role != 'decode' or not cfg.get('credit_fast_lane'):
        return
    assert cfg.get('dynamic_layer_extension'), 'Qualify only the existing credit-aware worker'
    import vllm
    from ..dense_ar_slo_layer_pd import worker
    from ..dense_ar_slo_layer_pd.credits import LEDGER
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.block_pool import BlockPool
    mutex = threading.RLock()
    holder, channels, events = {}, [], []
    for name in ('map_request', 'outstanding', 'reserve', 'commit', 'cancel', 'validate', 'allocated', 'finish'):
        setattr(LEDGER, name, locked(mutex, getattr(LEDGER, name)))

    def handle(command):
        received = time.monotonic()
        with mutex:
            assert 'scheduler' in holder, 'Control arrived before the normal worker-ready response'
            scheduler = holder['scheduler']
            generation = command['generation']
            kind = command['kind']
            if kind == 'reserve':
                free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
                active = list(scheduler.requests.values())
                future = sum((max(0, (r.sampling_params.max_tokens if r.sampling_params else 0) - r.num_output_tokens) + 15) // 16 for r in active)
                accepted = LEDGER.reserve(generation, command['cut'], command['requests'],
                    command['blocks_each'], free, decode_reserve=future,
                    max_pending=cfg.get('max_reserved_requests', 16))
                response = dict(kind='reservation_result', request_id=generation, accepted=accepted,
                    cut=command['cut'], KV_free_blocks=free, reserved_blocks=LEDGER.outstanding(),
                    time=time.monotonic(), generation=generation)
            elif kind == 'commit':
                LEDGER.commit(generation)
                response = dict(kind='commit_result', request_id=generation, generation=generation,
                                accepted=True, time=time.monotonic())
            else:
                if generation in LEDGER.reservations:
                    LEDGER.cancel(generation)
                response = dict(kind='reservation_cancelled', request_id=generation,
                                generation=generation, time=time.monotonic())
            events.append(dict(received=received, completed=time.monotonic(), command=kind,
                               generation=generation, handler_thread=threading.get_ident()))
            return response

    original_client = worker.Client
    def client(*args, **kwargs):
        value = ControlChannel(original_client(*args, **kwargs), handle)
        channels.append(value)
        return value
    worker.Client = client
    llm_constructor = vllm.LLM
    def initialized(*args, **kwargs):
        llm = llm_constructor(*args, **kwargs)
        # The inherited worker installs its wrappers before constructing LLM.
        # Wrap those final methods, not an earlier allocator implementation.
        for name in ('schedule', 'update_from_output', 'add_request', 'finish_requests'):
            if hasattr(Scheduler, name):
                setattr(Scheduler, name, locked(mutex, getattr(Scheduler, name)))
        KVCacheManager.allocate_slots = locked(mutex, KVCacheManager.allocate_slots)
        for name in ('get_new_blocks', 'free_blocks', 'touch', 'cache_full_blocks'):
            if hasattr(BlockPool, name):
                setattr(BlockPool, name, locked(mutex, getattr(BlockPool, name)))
        holder['scheduler'] = llm.llm_engine.engine_core.engine_core.scheduler
        return llm
    vllm.LLM = initialized
    def export():
        write(ROOT / 'raw' / cfg['name'] / 'credit_fast_lane_decode.json', dict(events=events,
            errors=[repr(channel.error) for channel in channels if channel.error],
            evidence='MEASURED_HOST_CONTROL_BOUNDARIES',
            scope='One IPC reader. Serialized IPC sends. Shared RLock covers scheduler/allocator bookkeeping and the unchanged credit ledger. Control handles only reserve/commit/cancel; it calls no CUDA or NIXL operation. Normal completion is tested separately from failures/cancellation after commit.'))
    atexit.register(export)
