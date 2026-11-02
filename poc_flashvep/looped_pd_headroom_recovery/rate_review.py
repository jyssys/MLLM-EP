"""Preserve finite-pilot loading when choosing a new confirmation rate."""
import copy
import os
import signal
import subprocess
from .common import *
from .prepare import put


def children(pid):
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            values=dict(line.split(':',1) for line in (p/'status').read_text().splitlines() if ':' in line)
            if int(values['PPid'])==pid:result.append(int(p.name))
        except FileNotFoundError:pass
    return result


def main():
    receipt=ROOT/'state/confirmation_rate_review.json'
    if receipt.exists():return
    while True:
        state=ROOT/'state/confirmation.json'
        if state.exists() and json.loads(state.read_text()).get('stage')=='CONFIRMATION_PLAN_READY':break
        time.sleep(.2)
    path=ROOT/'state/confirmation_plan.json';original=json.loads(path.read_text());plan=copy.deepcopy(original)
    rates={};changes=[]
    for item in plan['selection']:
        if not item.get('PD_pilot'):continue
        c=json.loads((ROOT/'configs'/(item['PD_pilot']+'.json')).read_text())
        realized=(c['count']-1)/(c['arrival_offsets'][-1]-c['arrival_offsets'][0])
        new=c['arrival_rate']
        if abs(realized/new-1)>.15:new=max(.25,min(5. if c['model']=='ouro' else 2.,.25*round(realized/.25)))
        rates[c['model']]=new
        item.update(pilot_nominal_QPS=c['arrival_rate'],pilot_realized_interval_rate=realized,QPS=new,
            confirmation_rate_scope='New nominal confirmation point near the observed pilot loading. Not a claim that nominal and realized QPS are equal.')
    pid=89652
    proc=Path('/proc')/str(pid)
    command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert proc.stat().st_uid==os.getuid() and command.endswith('headroom_recovery.confirmation ')
    assert not children(pid),'Never interrupt a running native experiment'
    os.kill(pid,signal.SIGSTOP)
    try:
        assert json.loads(state.read_text()).get('stage')=='CONFIRMATION_PLAN_READY' and not children(pid)
        for item in plan['runs']:
            rate=rates[item['model']]
            for role in ('PD','DP'):
                old=item[role]
                if old is None:continue
                c=json.loads((ROOT/'configs'/(old+'.json')).read_text())
                if abs(c['arrival_rate']-rate)<1e-12:continue
                c.update(name=old.replace('_n128','_rate_matched_v2_n128'),rate_review_original_name=old,
                    rate_review_original_nominal_QPS=c['arrival_rate'])
                item[role]=put(c,128,rate,split='confirmation')
                changes.append(dict(original=old,replacement=item[role],new_nominal_QPS=rate))
        backup=ROOT/'state/confirmation_plan_pre_rate_review.json'
        if backup.exists():assert json.loads(backup.read_text())==original
        else:write(backup,original)
        plan.update(rate_review=dict(time=time.time(),declaration='state/confirmation_rate_review_rule.json',
            original_plan='state/confirmation_plan_pre_rate_review.json',changes=changes,
            scope='No N128 result exists when this rate is fixed. Both policies receive identical new Poisson schedules. All frozen SLOs and model settings remain unchanged. Original prepared configs remain unrun, not overwritten.'))
        write(path,plan)
        os.kill(pid,signal.SIGTERM)
    finally:
        try:os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:pass
    with (ROOT/'state/confirmation_rate_matched_v2.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([PYTHON,'-m','poc_flashvep.looped_pd_headroom_recovery.confirmation'],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(receipt,dict(time=time.time(),old_waiting_controller=pid,new_controller=child.pid,changes=changes,
        rule='Declared before N128 execution in confirmation_rate_review_rule.json. No active GPU process is signaled.'))
    print(json.dumps(dict(controller=child.pid,revised_configs=len(changes))))


if __name__=='__main__':main()
