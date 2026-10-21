"""Continue completed stages after the current GPU controller exits safely."""
import argparse
import os
import subprocess
import sys
import time
import traceback
from .contract import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--await-pid',type=int,required=True);a=p.parse_args()
    write(ROOT/'state/continuation_controller.json',dict(pid=os.getpid(),await_pid=a.await_pid,
        purpose='Experiment stage orchestration only. No serving scheduler or new prediction policy.'))
    try:
        while Path(f'/proc/{a.await_pid}').exists():
            if (ROOT/'state/PAUSE').exists():return
            time.sleep(5)
        from .effective_sources import sources
        assert not sources()['missing_cohorts'],'The canonical Static Single pilot is incomplete.'
        for label in ['quality_S4_pilot_v2','quality_L4_pilot_v1','quality_O4_pilot_v1','quality_T4_custom_pilot_v1']:
            assert read(ROOT/'commands'/f'{label}_completion.json')['code']==0,label
        stages=[('dynamic-probes',['-m','poc_flashvep.ouro_8arm.campaign','--stage','dynamic-probes']),
            ('dynamic-quality',['-m','poc_flashvep.ouro_8arm.campaign','--stage','dynamic-quality']),
            ('matched-natural',['-m','poc_flashvep.ouro_8arm.matched_natural']),
            ('freeze-dynamic-replay',['-m','poc_flashvep.ouro_8arm.freeze_dynamic_replay']),
            ('static-replay',['-m','poc_flashvep.ouro_8arm.replay_campaign']),
            ('eight-arm-depth-replay',['-m','poc_flashvep.ouro_8arm.replay_campaign','--dynamic']),
            ('extension400',['-m','poc_flashvep.ouro_8arm.extension400']),
            ('heavy-trace',['-m','poc_flashvep.ouro_8arm.replay_campaign','--trace'])]
        for name,args in stages:
            if (ROOT/'state/PAUSE').exists():return
            write(ROOT/'state/continuation_stage.json',dict(stage=name,pid=os.getpid(),status='RUNNING'))
            print('CONTINUATION',name,flush=True)
            subprocess.run([PYTHON,'-u']+args,cwd=PROJECT,check=True)
            if name in ['dynamic-quality','extension400']:
                for module in ['scoring','paired_analysis','serving_analysis','subset400_serving','work_audit','report_snapshot']:
                    subprocess.run([PYTHON,'-u','-m','poc_flashvep.ouro_8arm.'+module],cwd=PROJECT,check=True)
        profiles=sorted((ROOT/'profiles').glob('*.sqlite'))
        for profile in profiles:
            subprocess.run([PYTHON,'-u','-m','poc_flashvep.ouro_8arm.profile_analysis',str(profile)],cwd=PROJECT,check=True)
        for module in ['scoring','paired_analysis','serving_analysis','subset400_serving','replay_analysis','work_audit','observer_analysis','plots','serving_plots','validate_artifacts','report_snapshot']:
            subprocess.run([PYTHON,'-u','-m','poc_flashvep.ouro_8arm.'+module],cwd=PROJECT,check=True)
        write(ROOT/'state/continuation_stage.json',dict(stage='Raw experiments completed. Main-agent report review remains.',status='NEEDS_REVIEW'))
    except Exception as error:
        write(ROOT/'state/continuation_error.json',dict(error=str(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
