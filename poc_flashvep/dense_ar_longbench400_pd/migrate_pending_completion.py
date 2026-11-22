"""Move an unstarted whole A/B block. Never interrupt an active GPU job."""
import json
import os
from pathlib import Path
import signal
import time
from .common import ROOT,write,preserve,digest
from .run import save


def main():
    preserve()
    receipt=json.loads((ROOT/'state/background_completion_guard_discovery1_v1.json').read_text())
    pid=receipt['pid']
    process=Path('/proc')/str(pid)
    expected=b'poc_flashvep.dense_ar_longbench400_pd.prefill_completion_experiment\0discovery\x001\0'
    assert process.stat().st_uid==os.getuid() and expected in (process/'cmdline').read_bytes()
    original_path=ROOT/'configs/completion_guard_discovery_lane1_v1.json'
    original=json.loads(original_path.read_text())
    marker=ROOT/'state/completion_guard_resource_migration_v2.json'
    assert not marker.exists()
    terminated=False
    os.kill(pid,signal.SIGSTOP)
    try:
        for _ in range(100):
            status=(process/'status').read_text()
            if '\nState:\tT' in status:
                break
            time.sleep(.01)
        else:
            raise AssertionError('Confirm the owned controller stopped before checking its children')
        children=(process/'task'/str(pid)/'children').read_text().strip()
        begun=[name for name in original['jobs'] if (ROOT/'raw'/name).exists()]
        if children or begun:
            print('KEEP_ORIGINAL_COMPLETION_BLOCK_ALREADY_STARTED',children,begun,flush=True)
            return
        # A stopped controller with no child and no run directory cannot own
        # an active measurement. Terminate only this future-dispatch process.
        os.kill(pid,signal.SIGTERM)
        os.kill(pid,signal.SIGCONT)
        terminated=True
        for _ in range(100):
            if not process.exists() or not (process/'cmdline').read_bytes():
                break
            time.sleep(.02)
        else:
            raise AssertionError('Do not dispatch a second copy while the old controller can still run')
        names=[]
        mapping={}
        for old in original['jobs']:
            cfg=json.loads((ROOT/'configs'/(old+'.json')).read_text())
            assert cfg['gpus']==[4,5] and cfg['qps']==2.8
            name=old.removesuffix('_v1')+'_GPU23_v2'
            cfg.update(name=name,gpus=[2,3],resource_migration_from=old,
                resource_migration_scope='No original A/B GPU job started. Move the complete same-source/trace block to equivalent GPUs2/3. '
                    'Keep all four methods and both guard states on the same pair. Preserve the original unstarted plan and controller log. '
                    'No workload,SLO,action,source code or timing metric changes.')
            save(cfg)
            names.append(name)
            mapping[old]=name
        jobfile='completion_guard_discovery_q2_8_GPU23_v2.json'
        write(ROOT/'jobs'/jobfile,dict(jobs=names,stop_on_failure=True,
            original_frozen_plan_SHA256=digest(original_path),evidence='PRE_MEASUREMENT_RESOURCE_REASSIGNMENT'))
        write(marker,dict(time=time.time(),status='UNSTARTED_BLOCK_REASSIGNED',old_controller=receipt,
            stopped_only_future_dispatch=True,no_GPU_child_or_raw_job=True,old_pair=[4,5],new_pair=[2,3],
            job_manifest=jobfile,old_to_new=mapping,original_jobs_status='NOT_RUN_RESOURCE_REASSIGNMENT',
            scope='Use new names,not edits to frozen configs. All A/B arms move together before any measurement. '
                'Keep GPUs4/5 work untouched. This reuses an available equivalent pair after its credit sweep.'))
        preserve()
        print('UNSTARTED_COMPLETION_BLOCK_MOVED',jobfile,flush=True)
    finally:
        if not terminated and process.exists():
            os.kill(pid,signal.SIGCONT)


if __name__=='__main__':
    main()
