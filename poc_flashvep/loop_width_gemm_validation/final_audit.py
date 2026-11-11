"""Close GPU work and audit completed cells without hiding budget skips."""
import os
from .common import *


def main():
    initialize()
    from ..looped_pd_autonomous_headroom.common import resources
    owner=resources();wrappers=owner.burners()
    for row in owner.processes():
        proc=Path('/proc')/str(row['pid'])
        assert proc.stat().st_uid==os.getuid() and os.getsid(row['pid'])==wrappers.get(row['gpu']),('A non-Burner GPU process remains active',row)
    progress=json.loads((ROOT/'state/confirmation_progress.json').read_text())
    plan=json.loads((ROOT/'state/confirmation_selection.json').read_text())
    assert progress['closed']
    done={r['name']:r for r in progress['results'] if r['status']=='COMPLETE'}
    assert all(r.get('output_checked') for r in done.values())
    statuses=[]
    for name in plan['jobs']:
        path=ROOT/'raw'/name/'dispatcher_result.json'
        receipt=json.loads(path.read_text()) if path.exists() else {}
        status='COMPLETE' if name in done else receipt.get('status','NOT_RUN_AFTER_BUDGET_CUTOFF')
        statuses.append(dict(name=name,status=status,dispatcher=receipt))
    serving=json.loads((ROOT/'analysis/serving_results.json').read_text())
    measured=[r for r in serving if r.get('complete')]
    assert all(r['exact_output_work_lifetime_passed'] for r in measured)
    assert len([r for r in measured if r['count']==64])==15
    assert len([r for r in measured if r['count']==128 and r['model']=='huginn'])==24
    assert all(r['measured_graph_captures']==0 for r in measured)
    assert all(g.get('fallbacks',0)==0 and g.get('cold_entries',0)==0 for r in measured for g in r['measured_graph_counts'].values())
    inventory=json.loads((ROOT/'analysis/campaign_inventory.json').read_text())
    assert inventory['qualified_test_requests']==720 and inventory['all_recorded_qualifications_passed']
    assert inventory['all_packed_core_coverage_passed']
    serial=json.loads((ROOT/'analysis/serial_path_costs.json').read_text())
    assert len(serial)==31 and all(r.get('passed') for r in serial)
    write(ROOT/'state/confirmation_closeout.json',dict(time=time.time(),cells=statuses,
        completed=len(done),planned=len(plan['jobs']),all_completed_output_checked=True,
        serving_cells=len(measured),all_measured_exact=True,
        all_measured_prompt_graph_hits=True,
        scope='A budget skip is NOT_RUN. It is not a GPU, state, or performance failure.'))
    write(ROOT/'state/budget_interpretation.json',dict(time=time.time(),start_epoch=START,
        hard_deadline_epoch=DEADLINE,GPU_stop_epoch=GPU_DEADLINE,long_job_start_cutoff_epoch=LONG_START_CUTOFF,
        effective_maximum_hours=(DEADLINE-START)/3600,
        inherited_display_field_hours=json.loads((ROOT/'state/budget.json').read_text())['maximum_wall_hours'],
        scope='The inherited display field says 11 hours. The configured timestamps enforce six hours. The final two secondary Native controls did not start after the new-job cutoff.'))
    write(ROOT/'state/CAMPAIGN_CLOSED.json',dict(time=time.time(),scientific_GPU_work_closed=True,
        completed_confirmation_cells=len(done),planned_confirmation_cells=len(plan['jobs']),
        reason='All Huginn confirmation cells completed. The new-job cutoff prevented the last two secondary Ouro Native controls. No more GPU work will start in this namespace.'))
    print('SCIENTIFIC_GPU_WORK_CLOSED',len(done),len(plan['jobs']))


if __name__=='__main__':main()
