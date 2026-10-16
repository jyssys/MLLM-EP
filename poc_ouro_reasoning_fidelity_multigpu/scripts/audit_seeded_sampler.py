"""Record installed source boundaries. This audit does not execute GPU sampling."""
import ast
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def segment(path, name):
    source = path.read_text()
    nodes = [n for n in ast.walk(ast.parse(source))
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    assert len(nodes) == 1, (name, len(nodes))
    node = nodes[0]
    return dict(start_line=node.lineno, end_line=node.end_lineno,
                text=ast.get_source_segment(source, node))


def main():
    base = Path(PYTHON).parent.parent / 'lib/python3.14/site-packages/vllm'
    sampler = base / 'v1/sample/ops/topk_topp_sampler.py'
    runner = base / 'v1/worker/gpu_model_runner.py'
    batch = base / 'v1/worker/gpu_input_batch.py'
    forward = segment(sampler, 'forward_cuda')
    noise = segment(sampler, 'random_sample')
    assert 'or generators:' in forward['text']
    assert 'return self.forward_native(logits, generators, k, p)' in forward['text']
    assert 'q[i].exponential_(generator=generator)' in noise['text']
    runner_text = runner.read_text()
    assert 'generator.manual_seed(sampling_params.seed)' in runner_text
    result = dict(
        method='INSTALLED_SOURCE_AUDIT_ONLY', gpu_sampling_probe='NOT_RUN',
        files=[dict(path=str(p), sha256=sha(p)) for p in (sampler, runner, batch)],
        sampled_function_boundaries=dict(forward_cuda=forward, random_sample=noise),
        request_seed_initialization='A seeded request creates a device generator and applies sampling_params.seed.',
        actual_seeded_path='Nonempty request generators select the PyTorch-native sampler, not the FlashInfer branch.',
        noise='The native sampler draws exponential noise with each request generator.',
        startup_log_limit='The FlashInfer selection log does not identify the seeded request path.',
        observed_restart_evidence_sha256=sha(ROOT/'results/quality_engine_restart_reproducibility.json'),
        causal_attribution='UNRESOLVED. Source inspection does not measure the actual RNG offset or the first differing logits.',
        ruled_out_variant='The compared sessions have equal actual KV-pool geometry and equal engine arguments.',
        remaining_hypotheses=['Request-generator offsets or batching history differ.',
                              'Model or sampler arithmetic differs across sessions.'],
        no_main_run_setting_change=True)
    target = ROOT/'results/seeded_sampler_source_audit.json'
    assert not target.exists()
    write(target, result)
    print(result['actual_seeded_path'])


if __name__ == '__main__':
    main()
