"""Qualify exact continuous actions without extending an untested action family."""
import json
from .common import ROOT,write,digest
from .qualification import audit
from .quality_gate import compare


def main():
    results=[];actions=[];axes=[];full400=[]
    names=[f'C3Q6_{axis}4_Natural400_GPUdigest_v2' for axis in ('Layer','Token','Hybrid')]
    names += [p.stem for p in (ROOT/'configs').glob('C3Q10_*_Natural32_v1.json') if 'Heterogeneous' not in p.stem]
    for name in names:
        folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text())['complete']:continue
        cfg=json.loads((folder/'configuration.json').read_text())
        cp=ROOT/'analysis'/(name+'_coverage.json');qp=ROOT/'analysis'/(name+'_quality_gate.json')
        coverage=json.loads(cp.read_text()) if cp.exists() and json.loads(cp.read_text()).get('schema_version',0)>=3 else audit(name)
        quality=json.loads(qp.read_text()) if qp.exists() else compare(name)
        imported=coverage['imported_prefix_KV']
        passed=coverage['passed'] and coverage.get('decode_full36_observation_complete',False) and coverage['mixed_batches']>0 and quality['passed'] and len(imported)==cfg['count'] and all(r['status']=='PASS' for r in imported)
        row=dict(name=name,passed=passed,n=cfg['count'],mixed_batches=coverage['mixed_batches'],
            maximum_actual_D_batch=coverage['maximum_actual_D_batch'],quality={k:v for k,v in quality.items() if k!='rows'},
            all_valid_imported_prefix_KV_exact=all(r['status']=='PASS' for r in imported),
            source_SHA256=json.loads(receipt.read_text())['source_SHA256'])
        results.append(row)
        if passed:
            action=dict(axis=cfg['axis_kind'],cut=cfg['axis_cut'],suffix_fraction=cfg.get('hybrid_suffix_fraction') if cfg['axis_kind']=='Hybrid' else None)
            if action not in actions:actions.append(action)
            if cfg['axis_kind'] not in axes:axes.append(cfg['axis_kind'])
            if cfg['count']==400:full400.append(action)
    value=dict(passed=bool(actions),qualified_axes=axes,qualified_actions=actions,mixed_rows_passed=bool(actions),
        full400_actions=full400,results=results,analysis_SHA256=digest(__import__('pathlib').Path(__file__)),
        evidence='CORRECTNESS_ONLY',
        scope='Each exact action needs full valid imported-prefix hashes,full36 decode coverage,lifetimes,mixed rows,and frozen task-quality noninferiority. '
            'C0 full-state teachers independently validate the model DAG. '
            'Natural32 and Natural400 scopes remain distinct. This gate does not qualify decode safety or performance. '
            'Default-compile stock-model Token has its own gate and is not covered here.')
    write(ROOT/'state/C3_continuous_actions_gate_v1.json',value)
    print('Continuous actions qualified:',actions,'Full400:',full400,flush=True)
    return value


if __name__=='__main__':main()
