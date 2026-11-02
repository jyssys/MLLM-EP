"""Resume original-length recovery controls before the new workload search."""
import subprocess
import time
from .common import *


def wait_existing(name):
    folder=ROOT/'raw'/name
    while folder.exists() and not (folder/'dispatcher_result.json').exists():
        active=json.loads((ROOT/'state/active.json').read_text())
        if active.get('name')!=name or not Path('/proc',str(active.get('pid',0))).exists():
            raise RuntimeError('Orphan namespace requires review: '+name)
        time.sleep(5)


def run(name):
    wait_existing(name)
    result=ROOT/'raw'/name/'dispatcher_result.json'
    if result.exists():return json.loads(result.read_text()).get('complete',False)
    value=subprocess.run([str(PYTHON),'-m','poc_flashvep.looped_pd_headroom_recovery.run',name],cwd=REPO)
    return value.returncode==0


def gate(name,reference=None):
    configure()
    from ..looped_pd_atomic_prefill import evidence
    evidence.ROOT=ROOT
    value=dict(transfer=evidence.transfer(name),work=evidence.full_work(name))
    if reference:value['comparison']=evidence.compare(reference,name)
    value['passed']=all(v['passed'] for v in value.values())
    write(ROOT/'raw'/name/'recovery_gate.json',value)
    return value['passed']


def main():
    path=ROOT/'state/initial_plan_v3.json'
    if not path.exists():path=ROOT/'state/initial_plan_v2.json'
    plan=json.loads(path.read_text())
    ledger=[]
    first=plan['ouro_controls'][0]['gate']
    try:
        for item in plan['ouro_controls']:
            write(ROOT/'state/campaign.json',dict(stage='ORIGINAL_OURO_ADMISSION',next=item,time=time.time()))
            okay=run(item['gate'])
            okay=okay and gate(item['gate'],None if item['gate']==first else first)
            pilot=run(item['pilot']) if okay else False
            ledger.append(dict(model='ouro',label=item['label'],gate=okay,pilot=pilot))
            write(ROOT/'state/initial_results.json',ledger)
        name=plan['huginn_gate']
        write(ROOT/'state/campaign.json',dict(stage='ORIGINAL_HUGINN_PARITY',next=name,time=time.time()))
        hgate=run(name) and gate(name)
        ledger.append(dict(model='huginn',gate=hgate))
        write(ROOT/'state/initial_results.json',ledger)
        if hgate:
            for item in plan['huginn_profiles']:
                write(ROOT/'state/campaign.json',dict(stage='HUGINN_BOUNDED_P_ONLY_PROFILE',next=item,time=time.time()))
                clean=run(item['clean'])
                profile=run(item['profile']) if clean else False
                ledger.append(dict(model='huginn',**item,clean_complete=clean,profile_complete=profile))
                write(ROOT/'state/initial_results.json',ledger)
                if not profile:break
        write(ROOT/'state/campaign.json',dict(stage='INITIAL_CONTROLS_COMPLETE_REVIEW_REQUIRED',time=time.time(),results=ledger))
    except BaseException as error:
        write(ROOT/'state/campaign.json',dict(stage='CONTROLLER_FAILED_REVIEW_REQUIRED',time=time.time(),error=repr(error)))
        raise
    finally:
        # Another stage can reuse the quiet lane. If no registered work remains,
        # restore the user's resource policy instead of leaving idle devices.
        restore_if_idle()


if __name__=='__main__':
    main()
