"""Resume strong compiled-token discovery with the original baseline capsule."""
import json
from .common import ROOT,write
from .experiment import create
from .geometry import matched_suffixes
from .next_jobs import plan


def main():
    protocol=json.loads((ROOT/'configs/continuous_static_initial_discovery_v1.json').read_text())
    gate=json.loads((ROOT/'state/C3_stock_token_gate_v1.json').read_text())
    assert gate['full400_runtime_stress_passed']
    data=json.loads((ROOT/'data/balanced400_discovery.json').read_text());cells=[]
    for lane,(qps,first) in enumerate(((1.8,2),(2.8,6))):
        jobs=[]
        for cut in (2,4,8,12):
            assert dict(axis='Token',cut=cut,suffix_fraction=None) in gate['qualified_actions']
            suffixes,work=matched_suffixes([r['input_tokens'] for r in data],cut)
            name=create(f'C3Stock_Discovery200_q{qps}_Token{cut}_Resume_v1','Token','balanced400_discovery.json',
                cut=cut,pair=(first,first+1),qps=qps,seed=2026112291,
                execution_family='INSTALLED_DEFAULT_COMPILE',continuous_source_credit_limit=0,
                token_stock_model=True,qualification_gate='C3_stock_token_gate_v1.json',
                suffix_by_request={'q'+str(i):s for i,s in enumerate(suffixes)},aggregate_work_audit=work,
                source_capsule=protocol['source_capsule'],source_capsule_SHA256=protocol['source_capsule_SHA256'],
                discovery_only=True,
                scope='Same frozen Source,Discovery200,arrival seed,GPU pair,and default compiler as the original strong Native baseline. '
                    'These are normal continuous batches,not C0 batch-matched controls. '
                    'Cuts2,4,8,12 passed separate N8/N32 gates. Cut8 passed full400 runtime stress. '
                    'Do not use unqualified cut1 or treat this one-start tuning as a final win.')
            jobs.append(name);cells.append(dict(QPS=qps,cut=cut,name=name))
        plan(f'resume_strong_stock_token_lane{lane}_v1',jobs)
    write(ROOT/'configs/resume_strong_stock_token_discovery_v1.json',dict(cells=cells,
        source_capsule_SHA256=protocol['source_capsule_SHA256'],
        scope='Resume after the response-stream disconnect. Preserve every completed old run. No Dataset or SLO change.'))
    print('Frozen eight strong-token continuous discovery starts.',flush=True)


if __name__=='__main__':main()
