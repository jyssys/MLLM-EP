"""Select the expanded Huginn frontier only on the separate arrival selection trace."""
import argparse
import copy
from .common import *
from .analyze import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('declare','apply'));a=p.parse_args()
    initialize()
    path=ROOT/'state/earlier_frontier_selection_rule.json'
    if a.mode=='declare':
        assert not (ROOT/'raw/earlier_huginn_loop_selection32/measurement_begin.json').exists(), 'Declare before selection measurement'
        write(path,dict(time=time.time(),rule='Select the expanded candidate set only if its N32 Decode SLO screen passes and its TTFT p95 does not exceed the historical-cut Loop selection.',
            selection_arrival_seed=2026110920,confirmation_seeds=[2026110931,2026110932,2026110933],
            scope='Candidate-set selection only. This rule does not establish a serving benefit.'))
        return
    assert path.exists(),'The selection rule must precede selection measurement'
    q=json.loads((ROOT/'raw/earlier_huginn_dynamic_mixed_state32/qualification_with_frozen_reference.json').read_text())
    assert q['campaign_gate_passed']
    old=metrics('secondary_huginn_loop_selection32')
    new=metrics('earlier_huginn_loop_selection32')
    assert old['complete'] and new['complete']
    selected=new['Decode_SLO_screen'] and new['TTFT_p95_ms']<=old['TTFT_p95_ms']
    plan=json.loads((ROOT/'recovery_plan.json').read_text())
    if selected:
        for index,name in enumerate(plan['jobs']):
            if not name.startswith('recovery_huginn_loop_'):continue
            assert not (ROOT/'raw'/name).exists(),'Do not modify an executed confirmation'
            cfg=copy.deepcopy(json.loads((ROOT/'configs'/(name+'.json')).read_text()))
            cfg.update(name=name.replace('recovery_','recoveryx_',1),boundary_checkpoints=[16,21,24,28,30,31],
                required_qualification='earlier_huginn_dynamic_mixed_state32',
                selected_frontier_rule=str(path),phase='FROZEN_RECOVERY_EXTENDED_FRONTIER_CONFIRMATION')
            save(cfg);plan['jobs'][index]=cfg['name']
        plan['Huginn_expanded_frontier_selected']=True
        write(ROOT/'recovery_plan.json',plan)
    result=dict(time=time.time(),expanded_selected=selected,historical_selection=old,expanded_selection=new,
        scope='Only selection arrival data determine this choice. Confirmation data do not determine it.')
    write(ROOT/'state/earlier_frontier_selection_result.json',result)
    print(json.dumps(dict(expanded_selected=selected,old_TTFT_p95_ms=old['TTFT_p95_ms'],
        new_TTFT_p95_ms=new['TTFT_p95_ms'],new_Decode_SLO=new['Decode_SLO_screen'])))


if __name__=='__main__':
    main()
