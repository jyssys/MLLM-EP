"""Prepare one request at a time, then decode one fixed shared cohort."""
import hashlib
import inspect
import json
import subprocess
import sys
import textwrap
import time


def pending_prompt(request):
    return request.num_computed_tokens < request.num_prompt_tokens


def has_selected_prefill(scheduler, selected):
    return any(count > 0 and pending_prompt(scheduler.requests[rid])
               for rid, count in selected.items())


def defer_running(scheduler, request, selected):
    if pending_prompt(request):
        return has_selected_prefill(scheduler, selected)
    params = request.kv_transfer_params or {}
    cohort = params.get('dense_cohort_id')
    if cohort is None:
        return False
    released = getattr(scheduler, '_matched_prepared_cohorts', None)
    if released is None:
        released = scheduler._matched_prepared_cohorts = set()
    if cohort in released:
        return False
    members = [r for r in scheduler.requests.values()
               if (r.kv_transfer_params or {}).get('dense_cohort_id') == cohort]
    expected = params['dense_cohort_size']
    assert len(members) <= expected
    if len(members) != expected or any(pending_prompt(r) for r in members):
        return True
    released.add(cohort)
    return False


def transform_schedule(source):
    source = textwrap.dedent(source)
    running = '        request = self.running[req_index]\n'
    waiting = '        while (self.waiting or self.skipped_waiting) and token_budget > 0:\n'
    assert source.count(running) == source.count(waiting) == 1
    source = source.replace(running, running+
        '        if _matched_defer_running(self, request, num_scheduled_tokens):\n'
        '            req_index += 1\n'
        '            continue\n')
    return source.replace(waiting, waiting+
        '            if _matched_has_selected_prefill(self, num_scheduled_tokens):\n'
        '                break\n')


class FirstTokenBarrier:
    """Keep generation timestamps; count all collection time in client TTFT."""
    def __init__(self, channel, record):
        self.channel = channel
        self.record = record
        self.groups = {}
        self.by_request = {}

    def poll(self, *args):
        return self.channel.poll(*args)

    def recv(self):
        value = self.channel.recv()
        commands = value.get('commands', []) if value['kind']=='add_batch' else [value] if value['kind']=='add' else []
        for command in commands:
            rid = command['request_id']
            if not (rid.startswith('q') and rid[1:].isdigit()):
                continue
            params = command.get('kv_params') or {}
            cohort = params['dense_cohort_id']
            group = self.groups.setdefault(cohort, dict(expected=params['dense_cohort_size'],
                members=set(), first=set(), buffered=[], released=False))
            assert group['expected'] == params['dense_cohort_size']
            group['members'].add(rid)
            self.by_request[rid] = cohort
        return value

    def send(self, value):
        kind = value['kind']
        rid = value.get('request_id')
        cohort = self.by_request.get(rid)
        if cohort is None or kind not in ('output', 'prefill_token_ready'):
            return self.channel.send(value)
        group = self.groups[cohort]
        if group['released']:
            return self.channel.send(value)
        if not group['buffered']:
            group['hold_begin'] = time.monotonic()
        group['buffered'].append(value)
        if kind=='prefill_token_ready' or value.get('count')==1:
            group['first'].add(rid)
        if len(group['first']) == group['expected']:
            assert group['first'] == group['members']
            group['released'] = True
            self.record(dict(time=time.monotonic(), cohort=cohort,
                members=sorted(group['members']), hold_begin=group['hold_begin'],
                messages=len(group['buffered']), original_generation_timestamps_preserved=True,
                scope='Actual worker sends start only after every cohort first-token notice exists. Generation times are unchanged. This hold contributes to coordinator-visible TTFT.'))
            for message in group['buffered']:
                self.channel.send(message)
            group['buffered'].clear()

    def close(self):
        assert not any(g['buffered'] for g in self.groups.values())
        return self.channel.close()


def install(cfg, role):
    if role != 'decode':
        return
    from .common import ROOT, write
    from ..dense_ar_slo_layer_pd import worker as qualified
    original_client = qualified.Client
    path = ROOT/'raw'/cfg['name']/'matched_first_token_barrier.jsonl'
    def record(value):
        with path.open('a') as stream:
            stream.write(json.dumps(value)+'\n')
    qualified.Client = lambda *a, **kw:FirstTokenBarrier(original_client(*a, **kw), record)
    if cfg['evaluation_arm']=='Native':
        # A cached Native new-token query is Decode work. Keep its initial
        # full-cohort replay and original source-token reuse contract.
        return
    from vllm.v1.core.sched import scheduler as module
    old = inspect.getsource(module.Scheduler.schedule)
    code = transform_schedule(old)
    scope = dict(module.__dict__, _matched_defer_running=defer_running,
                 _matched_has_selected_prefill=has_selected_prefill)
    exec(compile(code, __file__+':single_prefill_fixed_cohort', 'exec'), scope)
    module.Scheduler.schedule = scope['schedule']
    write(ROOT/'raw'/cfg['name']/'matched_cohort_scheduler_source.json', dict(
        original_SHA256=hashlib.sha256(old.encode()).hexdigest(),
        executed_SHA256=hashlib.sha256(code.encode()).hexdigest(), source=code,
        scope='Only one prompt-prefill request per actual iteration. Preserve B8192, KV management, model math and full-cohort Decode. A ready cohort member cannot decode before the other prompt computations are submitted.'))


def cohort_limit(indices, cfg, maximum):
    size = cfg['matched_D_cohort_size']
    assert len(indices) >= size and maximum >= size
    return size


def transform_coordinator(source):
    needle = '    transformed = code.replace(needle, replacement)\n'
    extra = (
        "    group_gate = 'if pending_D and (not active_D or native_pass) and macro_ready:'\n"
        "    assert transformed.count(group_gate)==1\n"
        "    transformed = transformed.replace(group_gate, 'if pending_D and (not active_D or native_pass) and macro_ready and len(pending_D)>=cfg[\"matched_D_cohort_size\"]:')\n")
    assert source.count(needle) == 1
    source = source.replace(needle, needle+extra)
    source = source.replace(
        'Same FIFO, no-active-cohort and query-budget predicate in all PD arms. Actual cohort size can differ because query counts differ. No dynamic cut selection.',
        'Fixed FIFO cohort. Single-request prompt preparation spans multiple iterations before full-cohort Decode. Worker iteration budget remains8192. No dynamic cut selection.')
    return source


def coordinator_main():
    from . import serving
    source = transform_coordinator(inspect.getsource(serving.main))
    previous = subprocess.Popen
    class Popen(previous):
        def __init__(self, args, *a, **kw):
            if isinstance(args, (list, tuple)) and '-m' in args:
                args = list(args)
                index = args.index('-m')+1
                if args[index] in ('poc_flashvep.dense_ar_slo_layer_pd.native',
                                   'poc_flashvep.dense_ar_matched_axis_pd.worker'):
                    args[index] = 'poc_flashvep.dense_ar_matched_axis_pd.fixed_matched_cohort'
            super().__init__(args, *a, **kw)
    subprocess.Popen = Popen
    scope = dict(serving.__dict__, destination_limit=cohort_limit)
    exec(compile(source, __file__+':fixed_cohort_coordinator', 'exec'), scope)
    scope['main']()


def main():
    if '--role' not in sys.argv:
        return coordinator_main()
    from ..dense_ar_slo_layer_pd import native
    previous = native.stage
    def stage(cfg, role, address):
        install(cfg, role)
        return previous(cfg, role, address)
    native.stage = stage
    native.main()


if __name__ == '__main__':
    main()
