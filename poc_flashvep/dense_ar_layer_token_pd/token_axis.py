"""Transfer completed full-depth prefix KV, then execute the suffix on D."""
import atexit
import json
import time
from .common import ROOT, write


def suffix_for(cfg, request_id):
    external = request_id.split('-')[0]
    if cfg.get('token_suffix_by_request') and external.startswith('q') and external[1:].isdigit():
        return cfg['token_suffix_by_request'][int(external[1:])]
    return cfg['token_suffix']


def truncate_command(command, length, prefix):
    assert command['kind'] == 'add' and len(command['input_ids']) == length
    assert 0 < prefix < length and prefix % 16 == 0
    return dict(command, input_ids=command['input_ids'][:prefix])


def matched_token_plan(layers, count, group_size=4, length=8000):
    """Match GEMM token-layer rows exactly, not attention FLOPs or GPU time."""
    from fractions import Fraction
    target = Fraction(length * layers, 36)
    lower = (target.numerator // target.denominator // 16) * 16
    upper = lower + 16
    fraction = (target - lower) / 16
    assert count % group_size == 0
    groups = count // group_size
    assert groups % fraction.denominator == 0, 'Use a complete exact-budget period'
    plan = []
    for group in range(groups):
        selected = ((group + 1) * fraction.numerator // fraction.denominator >
                    group * fraction.numerator // fraction.denominator)
        plan.extend([upper if selected else lower] * group_size)
    assert 36 * sum(plan) == count * layers * length
    return plan


def install(cfg, role):
    if not cfg.get('token_axis'):
        return
    from ..looped_pd_atomic_prefill import ar_worker
    from ..dense_ar_slo_layer_pd import worker
    prefix = cfg['length'] - cfg['token_suffix']
    possible = cfg.get('token_suffix_by_request', [cfg['token_suffix']])
    assert all(0 < cfg['length'] - size < cfg['length'] and (cfg['length'] - size) % 16 == 0 for size in possible)
    assert not cfg.get('reuse_prefill_token') and not cfg.get('layer_mode')
    if cfg.get('dynamic_layer_extension'):
        assert cfg.get('policy') == 'NATIVE', 'Token-axis uses no partial-layer computation'
    folder = ROOT / 'raw' / cfg['name']
    records = []
    original_client = ar_worker.Client

    class PrefixChannel:
        def __init__(self, channel):
            self.channel = channel
        def poll(self, *args):
            return self.channel.poll(*args)
        def close(self):
            return self.channel.close()
        def recv(self):
            command = self.channel.recv()
            def transform(value):
                if role != 'prefill':
                    return value
                if value.get('kind') == 'add_batch':
                    return dict(value, commands=[transform(c) for c in value['commands']])
                if value.get('kind') == 'add':
                    size = suffix_for(cfg, value['request_id'])
                    actual_prefix = cfg['length'] - size
                    value = truncate_command(value, cfg['length'], actual_prefix)
                    records.append(dict(event='SOURCE_PREFIX_SUBMITTED', request_id=value['request_id'],
                        time=time.monotonic(), prefix_tokens=actual_prefix, suffix_tokens=size,
                        layers=36, rows=actual_prefix * 36))
                return value
            return transform(command)
        def send(self, message):
            if role == 'prefill' and message.get('kind') == 'output' and message.get('finished'):
                prefix = cfg['length'] - suffix_for(cfg, message['request_id'])
                params = message['kv_params']
                assert params['remote_num_tokens'] == prefix
                assert len(params['remote_block_ids'][0]) == prefix // 16
                message = dict(message, kv_params=dict(params, dense_token_prefix=prefix))
                records.append(dict(event='SOURCE_PREFIX_READY', request_id=message['request_id'],
                    time=message['time'], prefix_tokens=prefix, full_depth_prefix_KV_bytes=prefix * 36 * 2 * 8 * 128 * 2,
                    discarded_prefix_sample=message['output_ids']))
            return self.channel.send(message)

    def connected(*args, **kwargs):
        return PrefixChannel(original_client(*args, **kwargs))
    ar_worker.Client = connected
    # Instrumented worker imports the same Client symbol independently.
    worker.Client = connected
    if cfg.get('D_prefill_chunk'):
        def options_wrapper(original):
            def options(config, stage_role):
                value = original(config, stage_role)
                if stage_role == 'decode':
                    value['long_prefill_token_threshold'] = config['D_prefill_chunk']
                return value
            return options
        ar_worker.llm_options = options_wrapper(ar_worker.llm_options)
        worker.llm_options = options_wrapper(worker.llm_options)
    if role == 'decode':
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import NixlPullConnectorScheduler
        original_match = NixlPullConnectorScheduler.get_num_new_matched_tokens
        def match(self, request, computed):
            params = request.kv_transfer_params or {}
            if params.get('do_remote_prefill') and 'dense_token_prefix' in params:
                count = int(params['dense_token_prefix'])
                assert count == cfg['length'] - suffix_for(cfg, request.request_id)
                assert params['remote_num_tokens'] == count
                assert count < request.num_prompt_tokens
                records.append(dict(event='DESTINATION_PREFIX_MATCH', request_id=request.request_id,
                    time=time.monotonic(), imported_tokens=count, locally_computed=computed,
                    remaining_prompt_tokens=request.num_prompt_tokens - count, layers=36))
                return max(0, count - computed), count > computed
            return original_match(self, request, computed)
        NixlPullConnectorScheduler.get_num_new_matched_tokens = match
    atexit.register(lambda: write(folder / ('token_axis_' + role + '.json'), dict(records=records,
        evidence='STRUCTURAL_ACCOUNTING', scope='Actual submitted prefix, exported valid KV extent, and destination match. The P prefix sample is discarded. D computes all suffix tokens through all36 layers. No hidden state is transferred.')))


def install_state_witness(cfg, role):
    """Observe complete valid KV and the actual contiguous query rows."""
    if not cfg.get('axis_state_witness'):
        return
    import hashlib
    import torch
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM, Qwen3DecoderLayer
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    folder = ROOT / 'raw' / cfg['name']
    active, pending, completed, visits = {}, {}, {}, []
    prepare = GPUModelRunner.prepare_inputs
    def prepared(self, *args, **kwargs):
        batch = prepare(self, *args, **kwargs)
        active.update(batch=batch, runner=self, layers=[], graph_replay=False)
        return batch
    GPUModelRunner.prepare_inputs = prepared
    layer_forward = Qwen3DecoderLayer.forward
    def layer(self, *args, **kwargs):
        result = layer_forward(self, *args, **kwargs)
        if 'batch' in active:
            active['layers'].append(int(self.self_attn.attn.layer_name.split('.')[2]))
        return result
    Qwen3DecoderLayer.forward = layer
    def collect(value):
        batch = active.get('batch')
        if batch is None:
            return value
        runner = active['runner']
        for i, rid in enumerate(batch.req_ids):
            external = rid.split('-')[0]
            if not (external.startswith('q') and external[1:].isdigit()):
                continue
            idx = runner.req_states.req_id_to_index[rid]
            length = int(runner.req_states.prompt_len.np[idx])
            start = int(batch.num_computed_tokens_np[i])
            size = min(int(batch.num_scheduled_tokens[i]), length - start)
            visits.append(dict(request_id=external, start=start, queries=int(batch.num_scheduled_tokens[i]),
                prompt_length=length, layers=None if active.get('graph_replay') else list(active['layers']),
                graph_replay=bool(active.get('graph_replay'))))
            if size <= 0:
                continue
            row = int(batch.query_start_loc_np[i])
            state = pending.setdefault(rid, dict(cursor=start, first_query=start, hidden=[]))
            assert state['cursor'] == start, 'Missing or duplicate query work'
            state['hidden'].append(value[row:row + size].detach().clone().cpu())
            state['cursor'] += size
            if state['cursor'] != length:
                continue
            table = runner.block_tables.input_block_tables[0][i]
            positions = torch.arange(length, device=value.device)
            planes = []
            for number in range(36):
                attn = runner.vllm_config.compilation_config.static_forward_context[f'model.layers.{number}.self_attn.attn']
                valid = attn.kv_cache[table[positions // 16].long(), :, positions % 16, :].detach().contiguous().view(torch.uint8).cpu()
                planes.append(dict(layer=number, SHA256=hashlib.sha256(valid.numpy().tobytes()).hexdigest(), bytes=valid.numel()))
            hidden = torch.cat(state['hidden'])
            if cfg.get('axis_state_save_tensors', True):
                torch.save(hidden, folder / (f'axis_hidden_{role}_{external}.pt'))
            partition_hashes = {}
            if state['first_query'] == 0:
                for suffix in cfg.get('witness_hidden_partition_tokens', []):
                    boundary = length - suffix
                    if not 0 < boundary < len(hidden):
                        continue
                    partition_hashes[str(suffix)] = dict(
                        prefix_SHA256=hashlib.sha256(hidden[:boundary].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                        suffix_SHA256=hashlib.sha256(hidden[boundary:].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
            completed[external] = dict(KV=planes, first_query=state['first_query'], query_end=length,
                final_hidden_SHA256=hashlib.sha256(hidden.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                partition_hashes=partition_hashes, tensor_payload_saved=cfg.get('axis_state_save_tensors', True))
        return value
    if cfg.get('axis_post_runner_witness'):
        from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
        replay = CudaGraphManager.run_fullgraph
        def replayed(self, *args, **kwargs):
            active['graph_replay'] = True
            return replay(self, *args, **kwargs)
        CudaGraphManager.run_fullgraph = replayed
        execute = GPUModelRunner.execute_model
        def executed(self, *args, **kwargs):
            value = execute(self, *args, **kwargs)
            if kwargs.get('dummy_run') or torch.cuda.is_current_stream_capturing():
                return value
            frame = self.execute_model_state
            if frame is not None and frame.hidden_states is not None:
                active.update(batch=frame.input_batch, runner=self)
                collect(frame.hidden_states)
            return value
        GPUModelRunner.execute_model = executed
    else:
        forward = Qwen3ForCausalLM.forward
        def observed(self, *args, **kwargs):
            return collect(forward(self, *args, **kwargs))
        Qwen3ForCausalLM.forward = observed
    atexit.register(lambda: write(folder / ('axis_state_' + role + '.json'), dict(requests=completed,
        visits=visits, evidence='CORRECTNESS_ONLY', scope='CPU copies perturb timing. Valid prompt KV includes imported prefix and locally computed suffix. Hidden contains only actually executed queries. Layer visits include prompt and decode calls.')))
