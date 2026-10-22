"""Resume only unfinished profiler jobs, then rebuild analysis for human review."""
import os
import subprocess
import traceback
from .contract import *
from .lease import restore


def module(name,*args):
    print('RECOVERY',name,*args,flush=True)
    subprocess.run([PYTHON,'-u','-m','poc_flashvep.ouro_8arm.'+name,*args],cwd=PROJECT,check=True)


def main():
    write(ROOT/'state/recovery_controller.json',dict(pid=os.getpid(),purpose='Resume profiler exports and CPU analysis. Preserve completed quality and replay runs.'))
    try:
        for label in ['extension400_'+arm+'_v1' for arm in ARMS]+[f'native_TP{tp}_extension400_v1' for tp in [1,4]]:
            assert read(ROOT/'commands'/f'{label}_completion.json')['code']==0,label
            assert read(ROOT/'raw'/label/'complete.json')['success'],label
        write(ROOT/'state/continuation_stage.json',dict(stage='heavy-trace recovery',pid=os.getpid(),status='RUNNING'))
        module('replay_campaign','--trace','--resume')
        write(ROOT/'state/continuation_stage.json',dict(stage='CPU aggregation',pid=os.getpid(),status='RUNNING'))
        for profile in sorted((ROOT/'profiles').glob('*.sqlite')):module('profile_analysis',str(profile))
        for name in ['scoring','paired_analysis','serving_analysis','subset400_serving','replay_analysis','work_audit',
                     'behavior_analysis','gate_margin_analysis','placement_accounting','observer_analysis','plots',
                     'serving_plots','validate_artifacts','report_snapshot']:
            module(name)
        module('subset400_serving','--extension-only')
        write(ROOT/'state/continuation_stage.json',dict(stage='Raw experiments completed. Main-agent report review remains.',status='NEEDS_REVIEW'))
        module('archive_profiles')
    except Exception as error:
        write(ROOT/'state/recovery_error.json',dict(error=str(error),traceback=traceback.format_exc()))
        raise
    finally:restore([4,5,6,7])


if __name__=='__main__':main()
