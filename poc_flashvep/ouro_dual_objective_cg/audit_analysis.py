"""Controlled-state, event-observer, and selector-identity diagnostics."""
import json
from collections import defaultdict,deque
import numpy as np
from .common import ROOT,write
from .objective_families import choose
from ..ouro_budget_joint_cg.optimizer import Candidate
from ..ouro_budget_joint_cg.cost_predictor import csv_file

def main():
    alternatives=[];observer_pairs=[];residuals=[];identity=[]
    for folder in sorted((ROOT/'raw').glob('dual_p2_ALTERNATIVE_*')):
        if not (folder/'completed.json').exists():continue
        items={}
        for file in folder.glob('FORCED_*.json'):
            row=json.loads(file.read_text());items[(row['cell']['C'],row['cell']['G'])]=row
        for G in (1,2):
            small,large=items[(512,G)],items[(2048,G)]
            def gaps(data):
                return [1000*(b-a) for rid,clocks in data['token_times'].items() if rid!='target'
                    for a,b in zip(clocks,clocks[1:]) if a>=data['intervention_begin']]
            gs,gl=gaps(small),gaps(large)
            alternatives.append(dict(run_id=folder.name,G=G,physical_GPU=small['physical_GPU'],
                initial_state_exact=small['initial_state']==large['initial_state'],
                background_active_small=small['background_active'],background_active_large=large['background_active'],
                C512_target_TTFT_ms=1000*small['target_TTFT_s'],C2048_target_TTFT_ms=1000*large['target_TTFT_s'],
                TTFT_ratio_small_over_large=small['target_TTFT_s']/large['target_TTFT_s'],
                C512_post_start_BG_ITL_P99_ms=None if not gs else float(np.percentile(gs,99)),
                C2048_post_start_BG_ITL_P99_ms=None if not gl else float(np.percentile(gl,99)),
                both_R4=small['full_R4'] and large['full_R4'],
                scope='LIVE_MEASURED_FORCED_COMPONENT. Causal initial-state interpretation only when exact state hashes match. Not adaptive or full-answer gain.'))
    for folder in sorted((ROOT/'raw').glob('dual_p2_EVENT_*')):
        if not (folder/'completed.json').exists():continue
        records=[]
        for file in folder.glob('*.json'):
            data=json.loads(file.read_text())
            if isinstance(data,dict) and data.get('heavy_observer') is not None and 'label' in data:records.append(data)
        keyed={r['label']:r for r in records}
        for label,data in keyed.items():
            if not label.endswith('_OBS'):continue
            clean=keyed[label.removesuffix('_OBS')+'_NOOBS']
            a={r['id']:r for r in data['requests']};b={r['id']:r for r in clean['requests']}
            observer_pairs.append(dict(run_id=folder.name,policy=label.removesuffix('_OBS'),requests=len(a),
                token_exact=sum(a[k]['output_ids']==b[k]['output_ids'] for k in a),
                output_count_equal=sum(len(a[k]['output_ids'])==len(b[k]['output_ids']) for k in a),
                observer_over_clean_wall=(data['end']-data['start'])/(clean['end']-clean['start']),
                scope='Paired natural runs. Observer feedback can alter causal choices and outputs. Not a pure timing-overhead estimate.'))
            quanta=defaultdict(deque)
            for q in data['service_observations']:
                for s in q['spans']:
                    if s['phase'].startswith('prefill'):quanta[(s['id'],s['start'],s['depth'],s['phase'])].append(q)
            for d in data['optimizer_decisions']:
                state=d['state'];key=(state['id'],state['start'],state['depth'],state['phase'])
                if not quanta[key]:continue
                q=quanta[key].popleft();c=d['selected']
                prediction=c['opening_ms'] if state['phase']=='prefilling_prelude' else c['current_ms']
                residuals.append(dict(run_id=folder.name,policy=label,phase=state['phase'],C=c['C'],
                    actual_G=0 if state['phase']=='prefilling_prelude' else c['G'],planned_G=c['G'],
                    actual_service_ms=q['service_ms'],predicted_unit_ms=prediction,B_ms=d['budget_ms'],
                    absolute_error_ms=abs(q['service_ms']-prediction),actual_over_budget=q['service_ms']>d['budget_ms'],
                    known=c['known'],fallback='FALLBACK' in d['reason'],alignment_bound_ms=q['alignment_bound_ms'],
                    scope='LIVE_MEASURED_READY_STREAM_SERVICE. CALIBRATION audit, not clean serving or kernel-only busy time.'))
    for folder in sorted((ROOT/'raw').glob('dual_p2_VAL_*')):
        if not (folder/'completed.json').exists():continue
        for file in folder.glob('*_J_VAL_*.json'):
            data=json.loads(file.read_text())
            for i,d in enumerate(data['optimizer_decisions']):
                state=d['state'];selected=d['selected']
                c_fixed=state['C'] if state['kind']=='CONTINUE_CHUNK' else min(2048,state['prompt_remaining'],state['token_budget'])
                items=[(Candidate(**{k:c[k] for k in Candidate.__dataclass_fields__}),c['features']) for c in d['candidates']]
                restricted=[p for p in items if p[0].C==c_fixed]
                if not restricted or 'FALLBACK' in d['reason']:continue
                disabled,_=choose(d['family'],restricted,**d['objective_parameters'])
                same_C=selected['C']==c_fixed
                identity.append(dict(run_id=folder.name,scenario=data['scenario']['id'],policy=data['label'],decision=i,
                    kind=state['kind'],C_same_as_fixed=same_C,
                    G_equal_when_same_C=disabled.G==selected['G'] if same_C else None,
                    full_candidate_C=selected['C'],disabled_C=c_fixed,full_G=selected['G'],disabled_G=disabled.G,
                    scope='OFFLINE_SELECTION_IDENTITY_ON_OBSERVED_STATE. Not an alternative live outcome.'))
    csv_file(ROOT/'controlled_alternative_pairs.csv',alternatives)
    csv_file(ROOT/'observer_output_parity.csv',observer_pairs)
    csv_file(ROOT/'event_budget_residuals.csv',residuals)
    csv_file(ROOT/'no_C_change_identity.csv',identity)
    assert all(r['G_equal_when_same_C'] for r in identity if r['C_same_as_fixed'])
    summary=dict(alternative_pairs=len(alternatives),initial_state_exact_pairs=sum(r['initial_state_exact'] for r in alternatives),
        observer_request_pairs=sum(r['requests'] for r in observer_pairs),observer_token_exact=sum(r['token_exact'] for r in observer_pairs),
        measured_budget_units=len(residuals),same_C_identity_decisions=sum(r['C_same_as_fixed'] for r in identity),
        same_C_score_identity=True)
    write(ROOT/'audit_summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
