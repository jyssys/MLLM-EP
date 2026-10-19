"""Same-device native option recheck. No new candidate or changed cohort."""
import argparse
import datetime as dt
import subprocess
from poc_ouro_parallelism_deep_discovery.contract import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--size',choices=MODELS,required=True)
    args=parser.parse_args();mask=validate_mask();assert len(mask)==1
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    state=dict(start_utc=dt.datetime.now(dt.timezone.utc).isoformat(),physical_gpu=mask[0],jobs=[],
        protocol='Same frozen sixteen calibration items and8192cap. Same GPU, separate fresh processes. No held-out items.',
        question='Does the natural-output difference reproduce when the physical device is matched?',
        evidence_class='QUALITY_ONLY',candidate='NONE')
    for enabled,label in [(False,'recheck_off'),(True,'recheck_on')]:
        command=[PYTHON,'-m','poc_ouro_parallelism_deep_discovery.scripts.native_quality_pilot',
            '--size',args.size,'--tp','1','--runtime-label',label]
        if enabled:command.append('--async-scheduling')
        job=dict(command=command,start_utc=dt.datetime.now(dt.timezone.utc).isoformat(),async_scheduling=enabled)
        state['jobs'].append(job);write(output/'pair_status.json',state)
        result=subprocess.run(command,cwd=PROJECT)
        job.update(exit_code=result.returncode,end_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write(output/'pair_status.json',state)
        if result.returncode:raise SystemExit(result.returncode)
    state.update(success=True,end_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write(output/'pair_status.json',state)

if __name__=='__main__':main()
