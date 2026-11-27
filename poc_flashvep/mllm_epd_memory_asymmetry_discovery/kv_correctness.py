"""Observe real vLLM prompt KV before and after a passive peer tier hit."""

import json

from ..mllm_epd_h200_prefill_heavy_batch_matched import runtime as rt


def inspect(run, names):
    import torch
    from safetensors import safe_open

    torch.set_num_threads(2)
    result = {'evidence': 'CORRECTNESS_ONLY', 'roles': {}}
    for role in ('P', 'D'):
        probes = [rt.read(run / 'raw/llm_reference' / f'{name}_{role}' / 'manifest.json') for name in names]
        for probe in probes:
            rt.require(probe['snapshot'] and rt.sha(probe['tensor_file']) == probe['tensor_SHA256'],
                       'Original complete native KV snapshot required')
        counts = probes[-1]['events'][0]
        prefix = counts['start'] if role == 'P' else probes[0]['snapshot']['event']['start']
        exact_prefix, exact_full, l2, max_abs = [], [], [], []
        with safe_open(probes[0]['tensor_file'], framework='pt', device='cpu') as first, \
                safe_open(probes[1]['tensor_file'], framework='pt', device='cpu') as second:
            exact_ids = torch.equal(first.get_tensor('prompt_token_ids'), second.get_tensor('prompt_token_ids'))
            exact_mrope = torch.equal(first.get_tensor('full_prompt_mrope'), second.get_tensor('full_prompt_mrope'))
            for layer in range(28):
                field = f'prompt_KV_{layer}'
                left, right = first.get_tensor(field), second.get_tensor(field)
                exact_prefix.append(bool(torch.equal(left[:prefix], right[:prefix])))
                exact_full.append(bool(torch.equal(left, right)))
                delta = left.float() - right.float()
                l2.append(float((delta.norm() / left.float().norm().clamp_min(1e-10)).item()))
                max_abs.append(float(delta.abs().max().item()))
        logits_a, logits_b = (probe['logits'] for probe in probes)
        top1 = [left['token_ids'][0][0] == right['token_ids'][0][0] for left, right in zip(logits_a, logits_b)]
        shared_logits = []
        for left, right in zip(logits_a, logits_b):
            lmap = dict(zip(left['token_ids'][0], left['values'][0], strict=True))
            rmap = dict(zip(right['token_ids'][0], right['values'][0], strict=True))
            shared_logits.extend(abs(lmap[key] - rmap[key]) for key in lmap.keys() & rmap.keys())
            if left['token_ids'][0][0] != right['token_ids'][0][0]:
                break
        result['roles'][role] = {
            'first_forward_computed_tokens_cached_arm': counts['start'], 'compared_prefix_rows': prefix,
            'prompt_ids_exact': exact_ids, 'mrope_tensor_exact': exact_mrope,
            'mrope_delta_exact': probes[0]['snapshot']['mrope_delta'] == probes[1]['snapshot']['mrope_delta'],
            'bitwise_prefix_layers': exact_prefix, 'bitwise_full_layers': exact_full,
            'layer_relative_L2': l2, 'layer_max_abs': max_abs,
            'greedy_logits_agreement_by_step': top1,
            'captured_logits_step_counts': [len(logits_a), len(logits_b)],
            'max_abs_on_shared_top32_to_first_divergence': max(shared_logits, default=None),
            'snapshot_sha256s': [probe['tensor_SHA256'] for probe in probes],
        }
    result['scope'] = ('Serial single-request probes with original CUDA compute and CPU capture. The first request starts '
                       'with a cold P cache. The restored P prefix must be observed through external-cache counters. '
                       'Logit comparisons use only the overlap of captured top-32 support and stop at the first greedy divergence.')
    return result


def transport_checks(run, names, reused_rows):
    import torch
    from safetensors import safe_open

    torch.set_num_threads(2)
    result = {}
    for name in names:
        p, d = (rt.read(run / 'raw/llm_reference' / f'{name}_{role}' / 'manifest.json') for role in ('P', 'D'))
        received = d['snapshot']['event']['start']
        with safe_open(p['tensor_file'], framework='pt', device='cpu') as source, \
                safe_open(d['tensor_file'], framework='pt', device='cpu') as dest:
            result[name] = {'D_imported_rows': received, 'P_to_D_bitwise_layers': [bool(torch.equal(
                source.get_tensor(f'prompt_KV_{layer}')[:received], dest.get_tensor(f'prompt_KV_{layer}')[:received],
            )) for layer in range(28)]}
    paths = [rt.read(run / 'raw/llm_reference' / f'{name}_D' / 'manifest.json')['tensor_file'] for name in names]
    with safe_open(paths[0], framework='pt', device='cpu') as left, \
            safe_open(paths[1], framework='pt', device='cpu') as right:
        result['D_byte_identical_reused_prefix_rows'] = reused_rows
        result['D_byte_identical_reused_prefix_layers'] = [bool(torch.equal(
            left.get_tensor(f'prompt_KV_{layer}')[:reused_rows], right.get_tensor(f'prompt_KV_{layer}')[:reused_rows],
        )) for layer in range(28)]
    return result


def execute(run, roster, cohort, metrics_snapshot, numeric_metrics):
    rt.require(len(roster) >= 3, 'Use target followed by distinct large eviction inputs')
    rpc = rt.module('llm_probe_control').rpc
    names = ('correctness_cold', 'correctness_peer_hit')

    def capture(name):
        live = []
        try:
            for port, role in ((18841, 'P'), (18842, 'D')):
                rpc(port, 'llm_probe/begin', {'name': name + '_' + role, 'max_logits': 16})
                live.append(port)
            cohort(name, roster[:1], 1, 16, True)
        finally:
            for port in reversed(live):
                rpc(port, 'llm_probe/end', {})

    capture(names[0])
    cohort('correctness_evict', roster[1:], 1, 16, True)
    external = 'vllm:external_prefix_cache_hits_total'
    local = 'vllm:prefix_cache_hits_total'
    peer_load = 'vllm:kv_offload_total_bytes_total:CPU_to_GPU'
    before = numeric_metrics(metrics_snapshot())['prefill']
    capture(names[1])
    after = numeric_metrics(metrics_snapshot())['prefill']
    delta = {key: after.get(key, 0) - before.get(key, 0) for key in (external, local, peer_load)}
    outcome = inspect(run, names)
    outcome['P_external_hit_tokens_engine_counter'] = delta[external]
    outcome['P_local_hit_tokens_engine_counter'] = delta[local]
    outcome['E_to_P_physical_page_bytes_counter'] = delta[peer_load]
    outputs = [json.loads((run / 'raw' / f'{name}_requests.jsonl').read_text())['token_ids'] for name in names]
    outcome['complete_output_token_ids_exact'] = outputs[0] == outputs[1]
    outcome['transport'] = transport_checks(run, names, outcome['roles']['P']['compared_prefix_rows'])
    outcome['observed_peer_hit'] = (delta[external] > 0 and delta[peer_load] > 0
                                    and outcome['roles']['P']['compared_prefix_rows'] > 0)
    rt.write(run / 'analysis/correctness_full28.json', outcome)
    rt.require(outcome['observed_peer_hit'], 'Cache correctness probe did not actually load a peer prefix')
