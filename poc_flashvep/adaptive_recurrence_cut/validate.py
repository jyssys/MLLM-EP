"""Validate finished evidence. Never replace missing results with estimates."""
import gzip
import json
from .common import ROOT,write
from .gate import read


def main():
    checks=[];failures=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        status=folder/'status.json'
        if not status.exists():failures.append(dict(name=folder.name,reason='Missing terminal status'));continue
        state=json.loads(status.read_text())
        if state['status']=='FAILED':
            failures.append(dict(name=folder.name,reason=state.get('error')));continue
        if state['status']!='COMPLETE':raise RuntimeError('Unfinished GPU run: '+folder.name)
        cfg=json.loads((folder/'arc_contract.json').read_text())
        rows=[r for r in read(folder/'requests.jsonl') if not r['warmup']]
        required=cfg['count']*len(cfg.get('cohorts',[cfg]))
        exact=len(rows)==required and all(all(d==(4 if cfg['model']=='ouro' else 32) for d in r['exit_depths']) for r in rows)
        if not cfg.get('cohorts'):
            before=json.loads((folder/'before_measured.json').read_text())
            after=json.loads((folder/'after_measured.json').read_text())
            issued=sum(after[k]['issued_loop_rows']-before[k]['issued_loop_rows'] for k in before)
            logical=sum((r['input_tokens']+r['output_tokens']-1)*r['recurrent_depth'] for r in rows)
            exact=exact and issued==logical
        checks.append(dict(name=folder.name,expected_measured_requests=required,actual=len(rows),passed=exact))
        if not exact:raise RuntimeError('Request or depth accounting failed: '+folder.name)
    data=[json.loads(gzip.decompress((ROOT/'data'/f'W1_{part}.json.gz').read_bytes())) for part in ('calibration','test')]
    hashes=[{r['prompt_sha256'] for r in values} for values in data]
    valid=(len(data[0])==len(data[1])==200 and not hashes[0].intersection(hashes[1])
           and all(len(r['input_ids']['huginn'])+r['controlled_output_cap']<=4096 for values in data for r in values))
    if not valid:raise RuntimeError('Long manifest split or context validation failed')
    for name in ('map_validation.json','native_control_parity.json','final_source_preservation.json','unrelated_index_preservation.json'):
        result=json.loads((ROOT/'state'/name).read_text())
        if not result['passed']:raise RuntimeError('Failed evidence gate: '+name)
    write(ROOT/'state/final_evidence_validation.json',dict(passed=True,completed_runs=len(checks),
          measured_requests=sum(r['actual'] for r in checks),run_checks=checks,preserved_failures=failures,
          W1_prepared_not_executed=dict(calibration=200,test=200,prompt_hashes_disjoint=True,context_valid=True)))
    print('VALIDATED',len(checks),'COMPLETED_RUNS;',len(failures),'PRESERVED_FAILURES')


if __name__=='__main__':main()
