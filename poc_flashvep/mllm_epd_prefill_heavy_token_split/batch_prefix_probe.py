"""Read-only per-request prefix hashes after the original mixed-batch forward."""
import hashlib
import os
import re
import time

from .common import ROOT, write
from .forward_probe import gather_prompt_kv


def tensor_signature(value):
    import torch
    data = value.detach().cpu().contiguous()
    return dict(shape=list(data.shape), dtype=str(data.dtype),
        SHA256=hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest())


def request_base(request_id, role, contracts):
    match = re.fullmatch(r'chatcmpl-(.+):([PD])-[0-9a-f]+', request_id)
    assert match and match[2] == role, ('Unknown actual vLLM request ID', request_id, role)
    assert match[1] in contracts, ('Unregistered correctness request', request_id)
    return match[1]


def forward_context():
    from vllm.forward_context import get_forward_context
    return get_forward_context()


class BatchPrefixProbe:
    def __init__(self, worker, name, contracts, allow_decode=False, capture_full_prompt=False):
        assert re.fullmatch('[A-Za-z0-9_.-]+', name) and name not in ('.', '..')
        self.role = dict(prefill='P', decode='D')[os.environ['MLLM_EPD_ROLE']]
        assert contracts and len(contracts) <= 32
        self.contracts = contracts
        assert not (allow_decode or capture_full_prompt) or self.role == 'D'
        self.allow_decode = allow_decode
        self.capture_full_prompt = capture_full_prompt
        self.full_tensors = {}
        for base, row in contracts.items():
            assert re.fullmatch('[A-Za-z0-9_.-]+', base)
            assert set(row) == {'prefix_tokens', 'full_prompt_tokens'}
            assert 0 < row['prefix_tokens'] < row['full_prompt_tokens']
        self.runner = worker.model_runner
        assert len(self.runner.kv_caches) == 28
        self.folder = ROOT / 'raw/batch_prefix' / name
        assert not self.folder.exists()
        self.folder.mkdir(parents=True)
        self.events = []
        self.snapshots = {}
        self.current = None

    def begin(self):
        me = self
        self.slots = [(key, key in vars(self.runner), vars(self.runner).get(key))
            for key in ['execute_model', '_model_forward']]
        original_execute = self.runner.execute_model
        original_forward = self.runner._model_forward

        def execute(output, *args, **kwargs):
            me.current = output
            # InputBatch is updated inside the original execute_model. Read it only
            # from the subsequent original _model_forward call site.
            return original_execute(output, *args, **kwargs)

        def forward(*args, **kwargs):
            batch = me.runner.input_batch
            output = me.current
            context = forward_context()
            requests = []
            offset = 0
            for index, rid in enumerate(batch.req_ids):
                count = int(output.num_scheduled_tokens[rid])
                assert count > 0
                base = request_base(rid, me.role, me.contracts)
                contract = me.contracts[base]
                length = int(batch.num_prompt_tokens[index])
                start = int(batch.num_computed_tokens_cpu[index])
                expected = contract['prefix_tokens'] if me.role == 'P' else contract['full_prompt_tokens']
                phase = 'decode' if start >= length else 'prefill'
                assert length == expected
                assert (phase == 'decode' and me.allow_decode) or start + count <= length, (
                    'Subsequent decode requires an explicit correctness-only option', rid, start, count, length)
                requests.append(dict(base=base, request_id=rid, row=index, start=start,
                    query_tokens=count, prompt_tokens=length, query_offset=offset, phase=phase))
                offset += count
            assert offset == output.total_num_scheduled_tokens
            event = dict(index=len(me.events), actual_batch=len(requests), actual_query_tokens=offset,
                cudagraph_mode=str(context.cudagraph_runtime_mode),
                batch_descriptor=str(context.batch_descriptor), requests=requests)
            result = original_forward(*args, **kwargs)
            # These CPU copies occur only after the real forward and its NIXL
            # dependencies. They invalidate timing, but do not replace a tensor.
            for record in requests:
                rid = record['request_id']
                state = me.runner.requests[rid]
                start, count = record['start'], record['query_tokens']
                offset = record['query_offset']
                actual = kwargs['positions'][..., offset:offset + count]
                if record['phase'] == 'decode':
                    import torch
                    expected = torch.arange(start,start + count,dtype=state.mrope_positions.dtype).repeat(3,1)
                    expected += state.mrope_position_delta
                else:
                    expected = state.mrope_positions[..., start:start + count]
                record['positions'] = tensor_signature(actual)
                record['expected_positions'] = tensor_signature(expected)
                assert record['positions'] == record['expected_positions'], ('MRoPE query mismatch', record)
                base = record['base']
                prefix = me.contracts[base]['prefix_tokens']
                ready = start + count == prefix if me.role == 'P' else start == prefix
                if ready and base not in me.snapshots:
                    index = record['row']
                    table = batch.block_table[0]
                    number = int(table.num_blocks_per_row[index])
                    blocks = table.block_table.np[index, :number].tolist()
                    layers = []
                    for layer, cache in enumerate(me.runner.kv_caches):
                        assert cache.shape[1] == 4 and cache.shape[3] == 256
                        data = gather_prompt_kv(cache, blocks, prefix)
                        layers.append(dict(layer=layer, **tensor_signature(data)))
                    import torch
                    me.snapshots[base] = dict(request_id=rid, captured_event=event['index'],
                        prefix_tokens=prefix, prompt_tokens=record['prompt_tokens'],
                        prefix_ids=tensor_signature(torch.tensor(state.prompt_token_ids[:prefix], dtype=torch.int64)),
                        prefix_mrope=tensor_signature(state.mrope_positions[..., :prefix]),
                        mrope_delta=state.mrope_position_delta, layers=layers,
                        physical_block_ids=blocks, cache_block_size=int(table.block_size))
                if (me.capture_full_prompt and record['phase'] == 'prefill' and
                    start + count == me.contracts[base]['full_prompt_tokens']):
                    assert base not in me.full_tensors
                    table = batch.block_table[0]
                    number = int(table.num_blocks_per_row[record['row']])
                    blocks = table.block_table.np[record['row'], :number].tolist()
                    length = me.contracts[base]['full_prompt_tokens']
                    me.full_tensors[base] = {f'prompt_KV_{layer}':
                        gather_prompt_kv(cache,blocks,length).detach().cpu().contiguous().clone()
                        for layer,cache in enumerate(me.runner.kv_caches)}
            me.events.append(event)
            return result

        self.runner.execute_model = execute
        self.runner._model_forward = forward
        return dict(folder=str(self.folder), evidence='CORRECTNESS_ONLY',
            scope='Original mixed-batch forward. Post-forward CPU hashes invalidate timing.')

    def close(self):
        for key, existed, value in self.slots:
            if existed:
                setattr(self.runner, key, value)
            else:
                delattr(self.runner, key)
        full_files = {}
        if self.capture_full_prompt:
            from safetensors.torch import save_file
            from .common import digest
            for base,tensors in self.full_tensors.items():
                path = self.folder / f'{base}_full_prompt.safetensors'
                save_file(tensors,str(path))
                full_files[base] = dict(path=str(path),SHA256=digest(path))
        record = dict(time=time.time(), role=self.role, contracts=self.contracts,
            events=self.events, snapshots=self.snapshots, evidence='CORRECTNESS_ONLY',
            allow_decode=self.allow_decode,full_prompt_tensor_files=full_files,
            scope='Prefix hashes and actual per-request query positions. '
                'Not a suffix KV equivalence, task-quality, latency or active-decode safety result.')
        write(self.folder / 'manifest.json', record)
        return dict(folder=str(self.folder), snapshots=len(self.snapshots), events=len(self.events))
