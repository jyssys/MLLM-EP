"""Close new GPU admission only after the original matrices finish."""
import json
import subprocess
import time
from . import common
from .common import ROOT,REPO,PYTHON,write,preserve
from .monitor import snapshot
from .runtime_freeze import check


def main():
    current=snapshot()
    assert current['ready_for_final_analysis'] and not current['critical_failures'],current
    assert (ROOT/'state/final_analysis_ready.json').exists(),'The automatic monitor must finish its resource handoff first'
    assert (ROOT/'state/last2_supplement_cancelled.json').exists()
    assert not list((ROOT/'raw').glob('last2_*/dispatcher_result.json'))
    preserve();check()
    for module,args in [('source_audit',[]),('validate',['--final'])]:
        result=subprocess.run([str(PYTHON),'-m','poc_flashvep.dense_ar_matched_axis_pd.'+module,*args],cwd=REPO)
        assert result.returncode==0,module
    marker=ROOT/'state/CAMPAIGN_CLOSED.json'
    if not marker.exists():
        write(marker,dict(time=time.time(),status='CLOSED',
            reason='The original Last4/8/12 matrices and prespecified controls completed. Close new experiment admission before resource verification and publication. Campaign completion does not imply a positive research result.',
            discovery_starts=232,confirmation_starts=120,admission_control_starts=52,
            Last2_sweep_cancelled=True,Last2_sweep_GPU_starts=0))
    target=ROOT/'state/final_GPU_resource_restoration.json'
    if target.exists():
        assert json.loads(target.read_text())['passed']
    else:
        common.bind_legacy()
        from ..dense_ar_layer_token_pd import resource_closeout as proven
        for key in ('ROOT','QUIET_LOCK','resources','preserve','write','digest'):
            setattr(proven,key,getattr(common,key))
        proven.main()
    preserve();check()
    write(ROOT/'state/science_closeout.json',dict(time=time.time(),status='GPU_WORK_CLOSED_WITH_EVIDENCE_LIMITATIONS',
        all_original_completion_flags=current['flags'],all_state_qualification=current['qualification'],
        resource_restoration_passed=True,
        scope='No additional Last2 GPU sweep ran. Keep profiler failures, incomplete measurement claims and all negative results. Reports and publication remain separate steps.'))
    print('MATCHED_AXIS_GPU_CLOSEOUT_VERIFIED',flush=True)


if __name__=='__main__':
    main()
