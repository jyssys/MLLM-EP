"""Close only verified idle campaign proxies and verify the eight burners."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from .common import ROOT,digest,preserve,resources,write

CONTROLLERS=['interference_PROBE_C40R4','suffix_QUANTA_C41R3','ec_INLINE_CAUSAL_C44','batch_COUNTER_C46']
PROXY_MODULES={__package__+'.proxy',__package__+'.whole_deflection_proxy',__package__+'.interference_probe_proxy'}
MARKER='final_resource_closeout_v0.json'


def pending_proxy_work(state):
    pending=sum(int(state.get(key,0)) for key in ['encoder_pending','prefill_pending','decode_pending'])
    active=state.get('active_http_streams',{})
    assert isinstance(active,dict)
    return pending+sum(int(value) for value in active.values())


def controllers_closed():
    rows=[]
    for tag in CONTROLLERS:
        state=ROOT/'state'/f'{tag}_controller.json';value=json.loads(state.read_text());pid=int(value['pid'])
        assert not Path('/proc',str(pid)).exists(), 'A specific campaign controller is still live: '+tag
        complete=ROOT/'state'/f'{tag}_controller_complete.json';failure=ROOT/'state'/f'{tag}_failure.json'
        assert complete.exists() or failure.exists(),'A missing process is not a successful experiment: '+tag
        terminal=complete if complete.exists() else failure
        rows.append(dict(tag=tag,pid=pid,terminal_marker=str(terminal.relative_to(ROOT)),terminal_SHA256=digest(terminal)))
    return rows


def require_final_closeout():
    record=json.loads((ROOT/'state'/MARKER).read_text());assert record['passed'] and record['scope']=='GPU_RESOURCE_CLOSEOUT'
    assert all(not Path('/proc',str(row['pid'])).exists() for row in record['controllers'])
    controllers_closed()
    return record


def close_idle_proxies():
    closed=[];untouched=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            words=(proc/'cmdline').read_bytes().decode().split('\0')
            if '-m' not in words or words[words.index('-m')+1] not in PROXY_MODULES:continue
            pid=int(proc.name);module=words[words.index('-m')+1]
            assert os.getsid(pid)==pid and '--name' in words and '--port' in words
            name=words[words.index('--name')+1];port=int(words[words.index('--port')+1])
            stdout=(proc/'fd/1').resolve();assert ROOT/'logs' in stdout.parents
            config=ROOT/'configs'/f'proxy_{name}.json'
            assert config.exists(),'No matching campaign proxy config'
            args=json.loads(config.read_text())['args'];assert args['name']==name and int(args['port'])==port
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3) as response:health=json.load(response)
            assert health['name']==name
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/campaign_state',timeout=3) as response:state=json.load(response)
            assert pending_proxy_work(state)==0,'Pending proxy work remains'
            assert (proc/'cmdline').read_bytes().decode().split('\0')==words and os.getsid(pid)==pid
            os.killpg(pid,signal.SIGTERM)
            until=time.monotonic()+10
            while proc.exists() and time.monotonic()<until:time.sleep(.1)
            assert not proc.exists(),'An owned proxy did not exit. No SIGKILL escalation.'
            closed.append(dict(pid=pid,module=module,name=name,port=port,config_SHA256=digest(config),
                stdout=str(stdout),observed_idle_state=state,action='SIGTERM_VERIFIED_OWNED_IDLE_PROXY'))
        except (FileNotFoundError,ProcessLookupError):continue
        except Exception as error:
            untouched.append(dict(pid=int(proc.name),error=type(error).__name__+': '+str(error),action='UNVERIFIED_PROCESS_LEFT_UNTOUCHED'))
    return closed,untouched


def main():
    preserve();target=ROOT/'state'/MARKER;assert not target.exists();controllers=controllers_closed()
    leases=[]
    for path in (ROOT/'state').glob('server_*.json'):
        if path.name.endswith('.source_snapshot.json'):continue
        value=json.loads(path.read_text())
        assert value['finished'] and not Path('/proc',str(value['guardian'])).exists(),path
        leases.append(path.name)
    proxies,untouched=close_idle_proxies();owner=resources()
    approved=json.loads((ROOT/'state/environment_initial.json').read_text())
    assert digest('/home/esjung/vllm-ep/run_utilize.sh')==approved['burner_wrapper_SHA256']
    assert digest(owner.SCRIPT)==approved['burner_implementation_SHA256']
    owner.restore(list(range(8)))
    until=time.monotonic()+45;rows=[];burners={}
    while True:
        burners=owner.burners();rows=owner.processes()
        if set(burners)=={r['gpu'] for r in rows}==set(range(8)):break
        assert time.monotonic()<until,'Burner CUDA contexts are not ready. Do not touch unrelated GPU work.'
        time.sleep(.5)
    for row in rows:
        proc=Path('/proc',str(row['pid']))
        assert proc.stat().st_uid==os.getuid() and os.getsid(row['pid'])==burners[row['gpu']]
    samples=[]
    for _ in range(3):
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used',
            '--format=csv,noheader,nounits'],text=True)
        samples.append(dict(time=time.time(),GPUs=[dict(zip(['gpu','utilization_percent','used_MiB'],
            map(int,line.split(',')))) for line in output.splitlines()]))
        time.sleep(1)
    assert all(any(row['gpu']==gpu and row['utilization_percent']>0 for sample in samples for row in sample['GPUs']) for gpu in range(8))
    write(target,dict(time=time.time(),passed=True,scope='GPU_RESOURCE_CLOSEOUT',controllers=controllers,
        closed_GPU_leases=len(leases),approved_burner_workers=burners,actual_GPU_processes=rows,
        utilization_samples=samples,closed_owned_idle_proxies=proxies,unverified_processes_untouched=untouched,
        burner_command='bash /home/esjung/vllm-ep/run_utilize.sh',
        burner_script_SHA256=digest('/home/esjung/vllm-ep/run_utilize.sh'),
        preservation='User checkout/index and old frozen evidence verified before and after closeout.',
        limitation='GPU closeout does not declare scientific GO. Unverified CPU processes are explicitly left untouched.'))
    preserve();print('FINAL_GPU_BURNERS_0_TO_7_VERIFIED',len(proxies),len(untouched),flush=True)


if __name__=='__main__':main()
