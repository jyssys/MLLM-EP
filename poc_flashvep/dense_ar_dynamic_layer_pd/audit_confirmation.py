"""Check held-out evidence without changing execution or selection."""
import zipfile
from .common import *
from .analyze import records


def main():
    path=ROOT/'analysis/heldout_results.json'
    assert path.exists(),'Confirmation must finish before this audit'
    results=json.loads(path.read_text())['results']
    plan=json.loads((ROOT/'state/heldout_plan.json').read_text())
    assert len(results)==len(plan['jobs'])==58
    extra=ROOT/'analysis/exploratory_fraction50_results.json'
    if extra.exists():
        added=json.loads(extra.read_text())['results'];assert len(added)==5
        results+=added
    expected=json.loads((ROOT/'state/heldout_runtime_freeze.json').read_text())['hashes']
    selection=digest(ROOT/'configs/heldout_selection.json')
    rows=[]
    for run in results:
        name=run['name'];folder=ROOT/'raw'/name
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        receipt=json.loads((folder/'dispatcher_result.json').read_text())
        failures=[]
        if not (receipt['complete'] and receipt['normal_worker_exit']):failures.append('INCOMPLETE_RUNTIME')
        if not run['output_gate']['passed']:failures.append('OUTPUT_MISMATCH')
        if cfg['selection_SHA256']!=selection:failures.append('SELECTION_CHANGED')
        if cfg.get('execution_events') or cfg.get('physical_activity'):failures.append('PRIMARY_PROFILER_ENABLED')
        if cfg['dataset']!='heldout.json':failures.append('WRONG_PROMPT_POOL')
        if len(records(folder/'requests.jsonl'))!=128:failures.append('REQUEST_COUNT')
        summaries=list(folder.glob('NVML-summary-*.json'))
        if not summaries or not all(json.loads(p.read_text()).get('isolation_passed') is True for p in summaries):
            failures.append('OWNERSHIP_OBSERVER_NOT_EXPLICIT_PASS')
        lifetimes=list(folder.glob('*lifetime_*.json'))
        if not lifetimes or not all(json.loads(p.read_text()).get('passed') is True for p in lifetimes):
            failures.append('NORMAL_LIFETIME_NOT_PASS')
        with zipfile.ZipFile(folder/'source.zip') as archive:
            differences=[p for p,h in expected.items() if p not in archive.namelist() or
                hashlib.sha256(archive.read(p)).hexdigest()!=h]
        if differences:failures.append('FROZEN_EXECUTION_SOURCE_CHANGED')
        rows.append(dict(name=name,passed=not failures,failures=failures,
            runtime_source_differences=differences,checked_runtime_files=len(expected),
            lifetime_files=[p.name for p in lifetimes],source_archive_SHA256=digest(folder/'source.zip')))
    value=dict(results=rows,passed=all(r['passed'] for r in rows),expected_primary_jobs=len(plan['jobs']),
        observed_jobs=len(results),evidence='ARTIFACT_AND_HOST_PROTOCOL_AUDIT',
        scope='Output parity, frozen execution source, explicit ownership observations, and normal lifetime checks. Full KV/Hidden tensor comparisons and per-layer visits belong to the separate qualification runs. This audit does not establish physical GPU slack or failure recovery.')
    write(ROOT/'analysis/heldout_evidence_audit.json',value)
    print('HELDOUT_EVIDENCE_AUDIT',len(rows),value['passed'])
    assert value['passed'],[(r['name'],r['failures']) for r in rows if not r['passed']]


if __name__=='__main__':main()
