"""Extract real operand and kernel evidence. Structural comparisons stay labeled."""
import json
from collections import Counter
from .common import ROOT, write


def main():
    evidence = []
    kernels = []
    for folder in sorted((ROOT/'raw').glob('diag256*')):
        for path in folder.glob('fusion_probe_*.json'):
            value = json.loads(path.read_text())
            by_id = {b['id']:b for b in value['batches']}
            for call in value['calls']:
                batch = by_id.get(call['batch_id'])
                if batch is None or not {s['phase'] for s in batch['spans']} >= {'decode', 'prefill'}:
                    continue
                example = dict(run=folder.name, worker_file=path.name, batch_id=batch['id'],
                               logical_rows=batch['rows'], spans=batch['spans'],
                               physical_linear_shapes=call['mm_shapes'],
                               cuda_event_envelope_ms=call['cuda_event_envelope_ms'],
                               evidence='MEASURED_DIAGNOSTIC',
                               scope='Real shared operands, not a replay or an E2E speedup')
                evidence.append(example)
        for path in folder.glob('mixed_core_trace_*.json'):
            events = json.loads(path.read_text())['traceEvents']
            gpu = [e for e in events if e.get('cat') == 'kernel']
            names = Counter(e['name'] for e in gpu)
            linear = [e for e in events if e.get('name') == 'aten::linear' and e.get('ph') == 'X']
            cpu = {e.get('args',{}).get('External id'):e for e in events if e.get('cat') == 'cpu_op'}
            categories = Counter()
            for event in gpu:
                category = 'other_kernels'
                node = cpu.get(event.get('args',{}).get('External id'), {})
                dims = node.get('args',{}).get('Input Dims', [])
                if 'flash' in event['name'].lower():
                    category = 'attention_kernel'
                elif node.get('name') in ('aten::mm','aten::addmm') and len(dims) >= 2:
                    matrices = [shape for shape in dims if len(shape) == 2]
                    if len(matrices) >= 2:
                        k, n = matrices[-2][-1], matrices[-1][-1]
                        if k in (5632,17920) or n in (5632,35840):
                            category = 'FFN_GEMM'
                        elif k == 10560:
                            category = 'reinjection_adapter_GEMM'
                        elif n == 1:
                            category = 'gate_projection'
                        else:
                            category = 'attention_projection_GEMM'
                categories[category] += event.get('dur',0)/1000
            total_kernel_ms = sum(categories.values())
            kernels.append(dict(run=folder.name, trace=path.name,
                                kernel_events=len(gpu), linear_operator_events=len(linear),
                                GEMM_GEMV_kernel_events=sum(count for name,count in names.items()
                                                           if any(v in name.lower() for v in ('nvjet','gemm','gemvx','xmma'))),
                                cuda_kernel_duration_sum_ms=sum(e.get('dur',0) for e in gpu)/1000,
                                kernel_duration_categories_ms=dict(categories),
                                kernel_duration_shares_pct={k:100*v/total_kernel_ms for k,v in categories.items()},
                                kernel_names=dict(names),
                                evidence='MEASURED_HEAVY_PROFILE',
                                limitation='One selected mixed core. Kernel durations are not an exclusive E2E partition.'))
    write(ROOT/'operator_fusion_evidence.json', dict(mixed_core_calls=evidence, selected_kernel_traces=kernels))
    print([(v['run'], v['batch_id'], v['logical_rows']) for v in evidence[:10]])
    print([(v['run'], v['kernel_events'], v['linear_operator_events']) for v in kernels])


if __name__ == '__main__':
    main()
