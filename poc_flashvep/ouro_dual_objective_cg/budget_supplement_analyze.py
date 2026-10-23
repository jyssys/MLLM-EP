"""Event-timed secondary budget units. Never classify these as clean serving."""
import json
from collections import defaultdict,deque
import numpy as np
from .common import ROOT
from ..ouro_budget_joint_cg.cost_predictor import csv_file

def main():
    rows=[];cells=[]
    for folder in sorted((ROOT/'raw').glob('dual_p2_BUDGET_*')):
        if not (folder/'completed.json').exists():continue
        for file in folder.glob('*.json'):
            data=json.loads(file.read_text())
            if not isinstance(data,dict) or 'scenario' not in data or 'label' not in data:continue
            assert data['heavy_observer']
            cells.append(dict(run_id=folder.name,policy=data['label'],B_ms=data['cell'].get('budget_ms'),
                width=data['configured_N'],offered=len(data['requests']),full_R4=data['full_R4'],
                natural_caps=sum(str(r['finish_reason']).split('.')[-1].lower() not in ('stop','eos','frontend_stop') for r in data['requests']),
                observer_units=len(data['service_observations']),progress_escapes=len(data['overbudget_admission']),
                scope='LIVE_MEASURED_EVENT_DIAGNOSTIC. Includes observer/feedback effects. Not a clean serving comparison or accuracy result.'))
            quanta=defaultdict(deque)
            for q in data['service_observations']:
                for s in q['spans']:
                    if s['phase'].startswith('prefill'):quanta[(s['id'],s['start'],s['depth'],s['phase'])].append(q)
            for decision in data['optimizer_decisions']:
                s=decision['state'];key=(s['id'],s['start'],s['depth'],s['phase'])
                if not quanta[key]:continue
                q=quanta[key].popleft();c=decision['selected'];B=decision['budget_ms']
                predicted=c['opening_ms'] if s['phase']=='prefilling_prelude' else c['current_ms']
                rows.append(dict(run_id=folder.name,policy=data['label'],B_ms=B,width=data['configured_N'],
                    phase=s['phase'],actual_C=c['C'],actual_G=0 if s['phase']=='prefilling_prelude' else c['G'],
                    planned_G=c['G'],predicted_ms=predicted,actual_ready_stream_ms=q['service_ms'],
                    absolute_error_ms=abs(q['service_ms']-predicted),over_B=q['service_ms']>B,
                    underestimated=q['service_ms']>predicted,known=c['known'],fallback='FALLBACK' in decision['reason'],
                    alignment_bound_ms=q['alignment_bound_ms'],evidence='LIVE_MEASURED_EVENT_SERVICE_NOT_KERNEL_ONLY_NOT_CLEAN_E2E'))
    groups=defaultdict(list)
    for r in rows:groups[(r['B_ms'],r['width'],r['policy'],r['phase'])].append(r)
    summary=[]
    for key,vs in groups.items():
        errors=[v['absolute_error_ms'] for v in vs]
        summary.append(dict(zip(('B_ms','width','policy','phase'),key),units=len(vs),
            MAE_ms=float(np.mean(errors)),P95_absolute_error_ms=float(np.percentile(errors,95)),
            over_B=sum(v['over_B'] for v in vs),underestimated=sum(v['underestimated'] for v in vs),
            fallback=sum(v['fallback'] for v in vs),known=sum(v['known'] for v in vs),
            independent_starts=1,scope='One exploratory process start per budget/context. Correlated units, no confirmatory CI or SLO guarantee.'))
    csv_file(ROOT/'secondary_budget_event_units.csv.gz',rows)
    csv_file(ROOT/'secondary_budget_event_summary.csv',summary);csv_file(ROOT/'secondary_budget_cells.csv',cells)
    print('Secondary budget diagnostics',len(cells),'cohorts',len(rows),'matched units')

if __name__=='__main__':main()
