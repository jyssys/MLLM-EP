"""Measure A/A jitter without changing the frozen observer acceptance gate."""
import time
from .prepare import *
from .run import execute


def main():
    prepare()
    while time.time()<LONG_START_CUTOFF:
        file=ROOT/'state/final_flush_confirmation.json'
        if file.exists() and json.loads(file.read_text())['status']=='COMPLETE':break
        time.sleep(10)
    results=[]
    for start in (1,2,3):
        for model in ('ouro','huginn'):
            cfg=configuration(model,f'a_{model}_heavy_off_AA_s{start}_n128',seed=2026110660+start)
            cfg.update(independent_start=start,phase='NATIVE_AA_NO_COLLECTOR_JITTER_CONTROL',
                paired_reference=f'a_{model}_heavy_off_force_s{start}_n128')
            save(cfg);complete=execute(cfg['name'])
            if complete:
                from .analyze import matched
                from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
                with quiet_analysis('Native_AA_local_tail_check'):
                    result=matched(cfg['name'],cfg['paired_reference'])
                results.append(result)
            else:results.append(dict(name=cfg['name'],complete=False))
            write(ROOT/'state/native_repeat_controls.json',dict(status='RUNNING',results=results))
    write(ROOT/'state/native_repeat_controls.json',dict(status='COMPLETE',results=results,
        scope='A/A controls diagnose run-to-run variation. They do not relax or override the five-percent observer gate.'))


if __name__=='__main__':main()
