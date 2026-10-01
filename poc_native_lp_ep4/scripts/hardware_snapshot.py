"""Read-only authorized-device condition snapshot; never manage foreign jobs."""
import argparse
import json
import subprocess
import time
from pathlib import Path

from poc_native_lp_ep4.contract import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    fields='index,uuid,name,memory.total,memory.used,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw'
    raw=subprocess.check_output(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True)
    rows=[dict(zip(fields.split(','),[x.strip() for x in line.split(',')])) for line in raw.splitlines()]
    # Unauthorized device activity is recorded as aggregate conditions only.
    atomic_json(a.output,{'host_ns':time.time_ns(),
        'authorized':[r for r in rows if r['index'] in {'0','1','4','5'}],
        'other_shared_fabric_utilization':{r['index']:r['utilization.gpu'] for r in rows if r['index'] not in {'0','1','4','5'}},
        'action':'READ_ONLY; other GPU processes never stopped or used'})
    print(a.output)


if __name__=='__main__':main()
