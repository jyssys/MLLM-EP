"""Run only the frozen output-work diagnostic when primary natural data exist."""
import json
import time
from .common import ROOT,write,digest,dispatch_allowed
from .c3_campaign import launch
from .c3_main import plan,analyze

def run():
    marker=ROOT/'state/c3_primary_knee_replay_complete_v5.json'
    priority=json.loads((ROOT/'configs/c3_budget_priority_amendment_v4.json').read_text());qps=priority['primary_confirmation_QPS']
    source=ROOT/'jobs/c3_heldout_serving_v2.json'
    while dispatch_allowed():
        if source.exists():
            parent=json.loads(source.read_text());blocks=[b for b in parent['blocks'] if b['qps']==qps]
            assert len(blocks)==3
            names=[n for b in blocks for n in b['arms'].values()]
            if all((ROOT/'analysis'/(n+'_metrics.json')).exists() for n in names):break
        time.sleep(5)
    else:
        write(marker,dict(time=time.time(),status='NOT_RUN',reason='Primary natural starts did not finish inside the dispatch budget'));return
    differences=[]
    for b in blocks:
        metrics={label:json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text()) for label,name in b['arms'].items()}
        for left,right in (('Layer','StockToken'),('Hybrid','StockToken'),('Layer','NativeStrong'),('Hybrid','NativeStrong'),('StockToken','NativeStrong'),('Deflection','NativeStrong')):
            a=metrics[left]['output_tokens'];c=metrics[right]['output_tokens'];difference=abs(a-c)/min(a,c)
            differences.append(dict(start=b['start'],left=left,right=right,actual_output_work_difference_fraction=difference))
    required=any(r['actual_output_work_difference_fraction']>.05 for r in differences)
    write(ROOT/'state/c3_primary_output_work_trigger_v5.json',dict(time=time.time(),required=required,threshold=.05,differences=differences,
        scope='Prospectively fixed work-mismatch trigger. Never replace natural-EOS results or use EOS as an online feature.'))
    if not required:
        write(marker,dict(time=time.time(),status='NOT_NEEDED',complete=True,required=False));return
    selection=json.loads((ROOT/'configs/c3_final_static_selection_v2.json').read_text());replay=plan(selection,heldout=True,replay=True)
    full=json.loads((ROOT/'jobs'/replay).read_text());chosen=[b for b in full['blocks'] if b['qps']==qps];lanes=[[],[],[],[]]
    for b in chosen:
        lane=b['pair'][0]//2
        lanes[lane]+=[name for name in b['arms'].values()]
    filename='c3_primary_knee_work_replay_v5.json'
    write(ROOT/'jobs'/filename,dict(time=time.time(),lanes=lanes,blocks=chosen,stop_on_failure=False,
        full_replay_plan=replay,full_replay_plan_SHA256=digest(ROOT/'jobs'/replay),
        scope='Three fresh heldout192 replay starts at the preselected Native knee. '
            'Use one fixed pair for all arms and starts. Future independent C1 EOS is controlled workload length,not an online input. '
            'Do not pool the natural and replay starts or claim a dynamic serving speedup.'))
    result=launch(filename,allow_partial_budget=True);analyze(replay)
    write(marker,dict(time=time.time(),status='COMPLETE' if result['complete'] else 'BUDGET_LIMITED_PARTIAL',
        complete=result['complete'],required=True,plan=filename,full_replay_plan=replay,evidence='DIAGNOSTIC_CONTROLLED_REPLAY'))

if __name__=='__main__':run()
