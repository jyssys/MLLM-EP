"""Summarize receipts without presenting pending jobs as completed work."""
from .common import *


def main():
    initialize();receipts=[];active=[]
    for path in sorted((ROOT/'raw').iterdir()):
        if not path.is_dir():continue
        p=path/'dispatcher_result.json'
        if p.exists():receipts.append(dict(name=path.name,**json.loads(p.read_text())))
        else:active.append(path.name)
    stages={}
    for name in ('load_refinement','colocated_layout_frontier','load_aware_colocated','static_validation',
        'wider_online_pilot','native_frontier','predictor_refinement','heldout','enclosure_controls'):
        path=ROOT/'state'/(name+'_progress.json')
        stages[name]=len(json.loads(path.read_text()).get('results',[])) if path.exists() else 0
    write(ROOT/'state/resume_manifest.json',dict(time=time.time(),preservation=preservation(),stage_completed_rows=stages,
        complete_jobs=sum(r['complete'] for r in receipts),failed_jobs=[r['name'] for r in receipts if not r['complete']],
        active_unreceipted_namespaces=active,closed=(ROOT/'state/CAMPAIGN_CLOSED.json').exists(),
        scope='Receipts describe runtime termination only. Output, state, profiler, and performance gates remain separate. Immutable configurations prevent accidental replay under a new meaning.'))
    print(json.dumps(dict(stages=stages,active=active,complete=sum(r['complete'] for r in receipts),failed=sum(not r['complete'] for r in receipts))))


if __name__=='__main__':main()
