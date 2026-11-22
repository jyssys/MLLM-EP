"""Record host pressure and device clocks without treating NVML as kernel activity."""
import json
from pathlib import Path
import subprocess
import time
from .common import ROOT,execution_allowed

def sample():
    fields='index,uuid,utilization.gpu,memory.used,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem,pstate'
    result=subprocess.run(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=15)
    pressure={name:Path('/proc/pressure',name).read_text() for name in ('cpu','io','memory')}
    return dict(time=time.time(),gpu_fields=fields,NVML_csv=result.stdout,query_returncode=result.returncode,
        query_error=result.stderr,host_pressure=pressure,load_average=Path('/proc/loadavg').read_text(),
        scope='Passive NVML and host-pressure observations. Utilization is not GPU kernel busy. '
            'Clocks and pressure alone do not establish the causal effect of shared-resource contention.')

def run():
    while execution_allowed():
        try:row=sample()
        except Exception as error:row=dict(time=time.time(),error=repr(error),evidence='DIAGNOSTIC')
        with (ROOT/'logs/environment_observations_v2.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        time.sleep(15)

if __name__=='__main__':run()
