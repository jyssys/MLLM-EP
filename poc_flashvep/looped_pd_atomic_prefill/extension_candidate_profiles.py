"""Use matched exploratory kernels to choose between Ouro input lengths."""
import fcntl
import subprocess
from .common import *
from .evidence import status
from .extension_discovery import enqueue
from .extension_runtime_controls import terminal

PLAN=ROOT/'state/extension_candidate_profiles_ouro_20261102.json'
RESULT=ROOT/'state/extension_candidate_profile_review_ouro_20261102.json'


def choose(options):
    if len({v['run']['length'] for v in options})<2:return max(options,key=lambda v:v['selection_score'])
    if not PLAN.exists():
        selected=[];jobs=[]
        for length in sorted({v['run']['length'] for v in options}):
            value=max((v for v in options if v['run']['length']==length),key=lambda v:v['selection_score'])
            name=value['run']['name'];cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            cfg.update(name=f'ext_candidate_profile_ouro_l{length}_v1',phase='profile_control',nsys=True,
                nsys_graph_trace='graph',profile_clean_reference=name,verify_transfer=False,state_probe=False,gpus=[0,1],timeout=1800,
                selection_profile_scope='Matched N24 exploratory profile. Select a length for independent qualification. Not final compute saturation or sustainable capacity.')
            jobs.append(cfg);selected.append(dict(length=length,clean=name,profile=cfg['name']))
        enqueue(jobs);write(PLAN,dict(time=time.time(),pairs=selected,stage='EXPLORATORY_PROFILES'));return None
    plan=json.loads(PLAN.read_text())
    if not terminal([v['profile'] for v in plan['pairs']]):return None
    if not RESULT.exists():
        if not plan.get('review_pid'):
            with (ROOT/'logs/extension_candidate_profile_review_ouro_20261102.log').open('a') as log:
                child=subprocess.Popen([str(PYTHON),'-m','poc_flashvep.looped_pd_atomic_prefill.extension_candidate_profiles'],
                    cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            plan.update(stage='CPU_REVIEW',review_pid=child.pid);write(PLAN,plan)
        return None
    review=json.loads(RESULT.read_text());qualified={}
    for item in review['profiles']:
        busy=next((d.get('kernel_busy_fraction') for d in item.get('devices',[]) if d['role']=='prefill'),None)
        qualified[item.get('clean_reference')]=bool(item.get('eligible_profile_transfer_to_clean') and busy is not None and busy>=.8)
    def score(value):
        activity=qualified.get(value['run']['name'],False);screen=all(value['checks'].values())
        return (screen and activity,screen,activity,*value['selection_score'])
    choice=max(options,key=score)
    choice['exploratory_profile_selection']=dict(profile_review=str(RESULT.relative_to(ROOT)),
        qualified_P_activity=qualified.get(choice['run']['name'],False),
        scope='N24 CUDA activity improves search selection only. Independent N32 and N128 gates remain unchanged.')
    plan.update(stage='SELECTION_COMPLETE',selected_clean=choice['run']['name'],
        selected_has_qualified_P_activity=qualified.get(choice['run']['name'],False))
    write(PLAN,plan)
    return choice


def analyze():
    from .profiles import parse
    plan=json.loads(PLAN.read_text());output=[]
    with (ROOT/'state/gpu_lease.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        for pair in plan['pairs']:
            name=pair['profile'];cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            if status(name)!='COMPLETE':
                output.append(dict(name=name,clean_reference=pair['clean'],status=status(name)));continue
            try:output.append(parse(ROOT/'raw'/name,cfg))
            except Exception as error:output.append(dict(name=name,clean_reference=pair['clean'],status='BLOCKED_PROFILE_REVIEW',error=repr(error)))
    write(RESULT,dict(time=time.time(),profiles=output,
        scope='Exploratory source-length selection. No N24 record can satisfy the final N128 cohort gate.'))


if __name__=='__main__':analyze()
