"""Compare scores on the same observed feasible workers. This is not a serving simulator."""
import json
from .common import ROOT
from .analyze import save_csv

def groups(rows):
 current=[];seen=set();identity=None
 for row in rows:
  key=(row['request_id'],row['role'])
  if current and (key!=identity or row['worker'] in seen):
   yield current;current=[];seen=set()
  identity=key;seen.add(row['worker']);current.append(row)
 if current:yield current

def main():
 output=[];detail=[]
 for path in sorted((ROOT/'raw').glob('*/shadow_states.jsonl')):
  status=path.parent/'status.json'
  if not status.exists() or json.loads(status.read_text()).get('status')!='COMPLETE':continue
  rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
  rows=[row for row in rows if not row['request_id'].startswith('r-1_')]
  for role in ('prefill','decode'):
   choices=[]
   for batch in groups(rows):
    if batch[0]['role']!=role:continue
    selected={label:min(batch,key=lambda row:tuple(row['actual_score'] if row['actual_policy']==label else row['scores'][label]))['worker']
     for label in ('queue','token','service','loop')}
    with_frontier=any(state.get('completed_prefill_frontiers') is not None for row in batch for state in row['states'])
    row=dict(run=path.parent.name,request_id=batch[0]['request_id'],role=role,feasible_workers=len(batch),
     queue_choice=selected['queue'],token_choice=selected['token'],service_choice=selected['service'],loop_choice=selected['loop'],
     service_loop_different=selected['service']!=selected['loop'],observed_frontier_available=with_frontier,
     evidence='MEASURED_ROUTING_STATES_WITH_COUNTERFACTUAL_SCORE_CHOICES_NOT_GPU_SPEEDUP')
    detail.append(row);choices.append(row)
   feasible=[r for r in choices if r['feasible_workers']>=2]
   informative=[r for r in feasible if r['observed_frontier_available']]
   output.append(dict(run=path.parent.name,role=role,scoring_rounds=len(choices),multi_worker_rounds=len(feasible),
    multi_worker_with_frontier=len(informative),service_loop_different=sum(r['service_loop_different'] for r in feasible),
    informative_service_loop_different=sum(r['service_loop_different'] for r in informative),
    confidence_interval='NOT_MEASURED_ONE_DIAGNOSTIC_START',
    evidence='MEASURED_ROUTING_OPPORTUNITIES_WITH_MODEL_PREDICTION_CHOICES',
    limitation='A scoring round is not necessarily an accepted admission. CPU logging changes scheduling. Scores are not future completion times.'))
 save_csv(ROOT/'C_shadow_opportunities.csv',output);save_csv(ROOT/'C_shadow_choices.csv',detail)

if __name__=='__main__':main()
