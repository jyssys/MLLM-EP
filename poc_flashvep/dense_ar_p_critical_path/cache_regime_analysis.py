"""Retain compiler-policy falsification and the separate warm-cache diagnostic."""
import json
import re
import time
from .common import ROOT,bind,write,digest
from .critical_analysis import csv_save


def analyze():
    bind();strict=ROOT/'state/fixed_port_stock_strict_complete_v6_reader_fix1.json'
    data=json.loads(strict.read_text());rows=[]
    for comparison in data['comparisons']:
        rows.append(dict(control=comparison['control'],candidate=comparison['candidate'],
            preplanned=True,requests=len(comparison['rows']),passed=comparison['passed'],
            KV_exact=sum(row['KV_exact'] for row in comparison['rows']),
            hidden_exact=sum(row['hidden_exact'] for row in comparison['rows']),
            greedy_exact=sum(row['greedy_exact'] for row in comparison['rows'])))
    diagnostic=ROOT/'analysis/fixed_port_warm_cache_pair_diagnostic.json'
    if diagnostic.exists():
        comparison=json.loads(diagnostic.read_text())['comparison']
        rows.append(dict(control=comparison['control'],candidate=comparison['candidate'],
            preplanned=False,requests=len(comparison['rows']),passed=comparison['passed'],
            KV_exact=sum(row['KV_exact'] for row in comparison['rows']),
            hidden_exact=sum(row['hidden_exact'] for row in comparison['rows']),
            greedy_exact=sum(row['greedy_exact'] for row in comparison['rows'])))
    paths=[]
    for name in (row['name'] for row in data['jobs']):
        for role in ('prefill','decode'):
            path=ROOT/'raw'/name/(role+'.log');log=path.read_text()
            paths.append(dict(name=name,role=role,log_SHA256=digest(path),
                cache_root=re.search(r'Using cache directory: (.*?) for vLLM',log)[1],
                new_graph_compilation='Compiling a graph for compile range ' in log,
                official_graph_reuse='Directly load the compiled graph(s)' in log,
                official_AOT_reuse='Directly load AOT compilation from path ' in log))
    warm=ROOT/'state/official_warm_cache_stock_strict_complete_v7.json'
    warm_result=json.loads(warm.read_text()) if warm.exists() else None
    qualification=ROOT/'configs/common_fixed_ports_qualification_v6.json'
    compositions=[]
    if qualification.exists():
        for item in json.loads(qualification.read_text())['items']:
            proof=json.loads((ROOT/'state'/item['gate']).read_text())
            names=[*(item.get('strict') or []),*([item['control']] if item.get('control') else []),*item['jobs']]
            completed=[name for name in names if (ROOT/'raw'/name/'dispatcher_result.json').exists()]
            compositions.append(dict(label=item['label'],composition_passed=proof['passed'],
                completed_GPU_controls=completed,not_run_GPU_controls=[name for name in names if name not in completed],
                prerequisite_failure='Required Stock cache/strict counterfactual failed' in proof.get('error',''),
                original_error=proof.get('error')))
    value=dict(time=time.time(),evidence='TRACE_ANALYSIS',strict_comparisons=rows,actual_compile_paths=paths,
        cache_roots_equal_by_role=data['cache_root_equal_by_role'],predeclared_V6_gate_passed=data['passed'],
        prospective_warm_regime=warm_result,composition_execution_audit=compositions,
        scope='Frozen port addresses removed the observed outer-cache fragmentation. They did not make the predeclared cold/warm state pairs exact. '
            'Both later warm-loaded runs matched each other in a retrospective CPU comparison. '
            'This supports a compile/load-path candidate,but does not prove the responsible operator. '
            'A new prospective warm triple is a separate correctness-only regime test. Do not pool it with prior numerical failures. '
            'No raw environment dictionary or credential is copied into this audit.')
    write(ROOT/'analysis/compiler_cache_regime_causal_audit.json',value)
    csv_save(ROOT/'figures/compiler_cache_regime_state_comparisons.csv',rows)
    csv_save(ROOT/'figures/compiler_cache_regime_compile_paths.csv',paths)
    print('COMPILER_REGIME_AUDIT',len(rows),'comparisons',flush=True)
    return value


if __name__=='__main__':analyze()
