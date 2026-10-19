"""Check the installed native DCP constraint. No vLLM, Torch, or CUDA import."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

from poc_ouro_parallelism_deep_discovery.contract import ROOT, MODELS, NATIVE_PYTHON, read, sha, write


def constraint_function(path):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ModelConfig')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'verify_with_parallel_config')
    blocks = [node for node in method.body if isinstance(node, ast.If)
              and 'decode_context_parallel_size > 1 and (not self.use_mla)' == ast.unparse(node.test)]
    if len(blocks) != 1:
        raise ValueError('The inspected native DCP validation block changed. Review its source before use.')
    arguments = ast.arguments(posonlyargs=[], args=[ast.arg(arg=name) for name in
        ['self', 'parallel_config', 'tensor_parallel_size', 'total_num_attention_heads',
         'decode_context_parallel_size']], kwonlyargs=[], kw_defaults=[], defaults=[])
    function = ast.FunctionDef(name='native_constraint', args=arguments,
                              body=blocks + [ast.Return(value=ast.Constant(value=True))], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    scope = {}
    exec(compile(module, str(path), 'exec'), scope)
    return scope['native_constraint'], dict(function=method.name, first_line=blocks[0].lineno,
        last_line=blocks[0].end_lineno, source_sha256=sha(path), extracted_code=ast.unparse(blocks[0]))


def check(fn, q_heads, kv_heads, tp, dcp):
    model = SimpleNamespace(use_mla=False, get_total_num_kv_heads=lambda: kv_heads)
    parallel = SimpleNamespace(decode_context_parallel_size=dcp)
    result = dict(query_heads=q_heads, KV_heads=kv_heads, TP=tp, DCP=dcp)
    try:
        fn(model, parallel, tp, q_heads, dcp)
        result['status'] = 'PASS_THIS_CONSTRAINT_ONLY'
    except AssertionError as error:
        result.update(status='REJECTED_BY_NATIVE_CONSTRAINT', reason=str(error))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    source = Path(NATIVE_PYTHON).parents[1] / 'lib/python3.14/site-packages/vllm/config/model.py'
    fn, provenance = constraint_function(source)
    rows, configs = [], []
    for size, model in MODELS.items():
        path = model['path'] / 'config.json'
        config = read(path)
        q, kv = config['num_attention_heads'], config['num_key_value_heads']
        assert q == kv == 16
        configs.append(dict(size=size, path=str(path), sha256=sha(path), query_heads=q, KV_heads=kv))
        for tp in [2, 4]:
            for dcp in [1, 2, 4]:
                if tp % dcp == 0:
                    rows.append(dict(size=size, **check(fn, q, kv, tp, dcp)))
    controls = [check(fn, 16, kv, 4, dcp) for kv, dcp in [(1, 2), (1, 4), (2, 2), (2, 4)]]
    assert all(row['status'] == ('PASS_THIS_CONSTRAINT_ONLY' if row['DCP'] == 1
                                else 'REJECTED_BY_NATIVE_CONSTRAINT') for row in rows)
    assert [row['status'] for row in controls] == ['PASS_THIS_CONSTRAINT_ONLY'] * 3 + ['REJECTED_BY_NATIVE_CONSTRAINT']
    write(output, dict(status='SOURCE_CONSTRAINT_REPRODUCED', source=str(source), provenance=provenance,
        model_configs=configs, target_rows=rows, positive_and_negative_controls=controls,
        scope='CPU execution of the exact inspected native eligibility block. Not full configuration validation or model execution.',
        GPU_execution=False, model_execution=False, candidate='NONE',
        conclusion='The installed native DCP>1 flag is ineligible for these MHA16 checkpoints at TP2/TP4. This is not a theorem against a different exact CP layout.'))
    print('NATIVE_DCP_SOURCE_CONSTRAINT', len(rows), 'target configurations;', len(controls), 'controls')


if __name__ == '__main__':
    main()
