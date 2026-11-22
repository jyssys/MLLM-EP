"""Checkpoint completed evidence without promoting discovery to confirmed wins."""
import json
import time
from .common import ROOT,write,digest,time_available
from .metrics import summarize
from .service_accounting import analyze
from .qualification import audit
from .quality_gate import compare as quality


def matrix():
    cells=[]
    for bucket in range(4):
        protocol=ROOT/f'configs/matrix_input{bucket}_protocol_frozen_v1.json'
        if not protocol.exists():continue
        for cell in json.loads(protocol.read_text())['cells']:
            for cut in (2,4,8,12):
                pair={}
                for axis in ('Layer','Token'):
                    name=f"MapC0_I{bucket}_O{cell['output']}_U{0 if cell['relative_load']==.65 else 1}_{axis}{cut}_v1"
                    path=ROOT/'analysis'/(name+'_metrics.json')
                    proof=ROOT/'analysis'/(name+'_coverage.json')
                    if path.exists() and proof.exists() and json.loads(proof.read_text())['passed']:
                        pair[axis]=json.loads(path.read_text())
                row=dict(input_bucket=bucket,output=cell['output'],cut=cut,QPS=cell['qps'],relative_load=cell['relative_load'],
                    independent_starts=1 if len(pair)==2 else 0,status='NOT_RUN',evidence='MEASURED_ONLINE_DISCOVERY')
                if len(pair)==2:
                    a,b=pair['Layer'],pair['Token']
                    row.update(status='TIE_OR_INCONCLUSIVE',reason='Only one start. No start-level confidence interval.',
                        Layer=a['name'],Token=b['name'],TTFT_p95_ratio=a['TTFT_p95_ms']/b['TTFT_p95_ms'],
                        SLO_goodput_ratio=a['SLO_qualified_output_tokens_per_second']/b['SLO_qualified_output_tokens_per_second'] if b['SLO_qualified_output_tokens_per_second'] else None,
                        ITL_p99_ratio=a['ITL_p99_ms']/b['ITL_p99_ms'],TPOT_p95_ratio=a['TPOT_p95_ms']/b['TPOT_p95_ms'])
                cells.append(row)
    write(ROOT/'analysis/axis_matrix_discovery_live.json',dict(time=time.time(),cells=cells,
        scope='All frozen input/output/cut/load cells. A measured one-start ratio does not confirm a winner. '
            'The same Native control is reused across cuts,not counted as multiple independent starts.'))


def once():
    errors=[];counts={'complete':0,'failed':0};code=digest(ROOT.parents[1]/'dense_ar_longbench400_pd/metrics.py')
    for path in sorted((ROOT/'raw').glob('*/dispatcher_result.json')):
        receipt=json.loads(path.read_text());name=path.parent.name
        if not receipt['complete']:counts['failed']+=1;continue
        counts['complete']+=1
        cfg=json.loads((path.parent/'configuration.json').read_text())
        try:
            cache=ROOT/'analysis'/(name+'_metrics.json')
            if not cache.exists() or json.loads(cache.read_text()).get('analysis_code_SHA256')!=code:
                result=summarize(name);result['analysis_code_SHA256']=code;write(cache,result)
            proof=ROOT/'analysis'/(name+'_coverage.json')
            if cfg.get('axis_kind') and (not proof.exists() or json.loads(proof.read_text()).get('schema_version',0)<3):audit(name)
            if cfg.get('controlled_pipeline') and not cfg.get('correctness_only') and not cfg.get('kernel_activity'):
                if not (ROOT/'analysis'/(name+'_service_accounting.json')).exists():analyze(name)
            if not cfg.get('ignore_eos') and not (ROOT/'analysis'/(name+'_quality_gate.json')).exists():quality(name)
        except Exception as error:
            errors.append(dict(name=name,error=repr(error),scope='Analysis failed. Do not infer a valid result from process exit alone.'))
    matrix()
    from .confirmation_analysis import main as confirm
    confirm()
    from .continuous_gate import main as continuous_gate
    continuous_gate()
    leases={path.stem:json.loads(path.read_text()) for path in (ROOT/'state').glob('pair_lease_*.json')}
    write(ROOT/'state/analysis_progress.json',dict(time=time.time(),status='ACTIVE',counts=counts,leases=leases,analysis_errors=errors,
        phase='AXIS_DISCOVERY_AND_CONTINUOUS_STATE_QUALIFICATION',
        scope='Checkpointed process and analysis status. This is not a final research decision.'))
    print(time.strftime('%Y-%m-%d %H:%M:%S'),counts,'analysis_errors',len(errors),flush=True)


def main():
    budget=json.loads((ROOT/'state/budget.json').read_text())
    while time_available() and not (ROOT/'state/STOP_ANALYSIS_WATCHER.json').exists():
        once();time.sleep(30)


if __name__=='__main__':main()
