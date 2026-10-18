"""Check the thin admission wrapper before starting the DP4 measurement lane."""
import json
import os
import subprocess
import time
from poc_decode_first_depth_aware_dp_v3.contract import *

def main():
    out=Path(os.environ['LOOPED_KV_OUTPUT'])
    rows=[r for r in read(ROOT/'artifacts/discovery_frozen.json')['rows'] if r['split']=='calibration'][:4]
    command=[PYTHON,'-m','poc_decode_first_depth_aware_dp_v3.scripts.live_worker',
        '--output',str(out/'rank'),'--batch','4','--trace']
    with (out/'worker_stderr.log').open('x') as log:
        proc=subprocess.Popen(command,cwd=PROJECT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=log,text=True,bufsize=1)
        def send(value):
            proc.stdin.write(json.dumps(value)+'\n');proc.stdin.flush()
        try:
            for line in proc.stdout:
                print(line,end='',flush=True)
                if not line.startswith('DP_EVENT '):continue
                value=json.loads(line[9:])
                if value['event']=='ready':send(dict(op='case',case='admission_smoke',origin_s=time.perf_counter()+.1))
                elif value['event']=='case_ready':
                    for row in rows:send(dict(op='request',row=row,cap=16,arrival_s=0))
                    send(dict(op='close'))
                elif value['event']=='case_done':send(dict(op='shutdown'))
            assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
    write(out/'status.json',dict(success=True,requests=len(rows),category='SINGLE_RANK_ADMISSION_WRAPPER_PROBE'))

if __name__=='__main__':main()
