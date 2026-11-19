"""Refresh final analyses under the same quiet-lane lease."""
import fcntl
import importlib
import time
from .common import ROOT,QUIET_LOCK,preserve,write


def main():
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    modules=('statistics','measurement_export','slot_decomposition','reservation_feedback','causal_decomposition','structural',
        'operator_analysis','calibration_sensitivity','evaluate_models','step_prediction_validation',
        'feature_validation','cross_cut_oracle','cohort_oracle_evaluation','control_replay_validation','figures','extended_figures')
    started=time.time();steps=[]
    preserve()
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        for name in modules:
            before=time.time()
            importlib.import_module('poc_flashvep.dense_ar_axis_sweetspot.'+name).main()
            steps.append(dict(module=name,elapsed_seconds=time.time()-before))
    preserve()
    write(ROOT/'analysis/final_analysis_pipeline_receipt.json',dict(time=time.time(),started=started,
        elapsed_seconds=time.time()-started,steps=steps,
        scope='No GPU job starts during this final refresh. Historical frozen replay/calibration artifacts are not regenerated. Figures and analyses contain only available measured cells and explicit replay models.'))
    print('FINAL_ANALYSIS_PIPELINE_COMPLETE',len(steps),flush=True)


if __name__=='__main__':
    main()
