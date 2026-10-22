"""Prospective shape subset, interleaved arms, five clean repeats, and separate traces."""
import argparse
import subprocess
from .contract import *
from .campaign import run
from .launch import execute
from .lease import restore


def capture_profile(label,argv,gpus,resume=False):
    profiles=ROOT/'profiles';chosen=label;attempt=1
    if resume:
        while (ROOT/'logs'/f'{chosen}.log').exists():
            completion=ROOT/'commands'/f'{chosen}_completion.json'
            marker=ROOT/'raw'/chosen/'complete.json'
            if completion.exists() and read(completion)['code']==0 and marker.exists() and read(marker).get('success'):
                break
            attempt+=1;chosen=label.rsplit('_v',1)[0]+f'_v{attempt}'
        if chosen!=label:
            mapping=ROOT/'state/profile_resume_map.json';values=read(mapping) if mapping.exists() else {}
            values[label]=chosen;write(mapping,values)
        argv=[chosen if word==label else str(profiles/chosen) if word==str(profiles/label) else word for word in argv]
    completion=ROOT/'commands'/f'{chosen}_completion.json'
    marker=ROOT/'raw'/chosen/'complete.json'
    completed=resume and completion.exists() and read(completion)['code']==0 and marker.exists() and read(marker).get('success')
    if not completed:execute(chosen,argv,gpus,timeout=1200,restore_after=False)
    sqlite=profiles/(chosen+'.sqlite')
    if not (resume and sqlite.exists()):
        subprocess.run(['nsys','export','--type=sqlite','--output',str(sqlite),str(profiles/(chosen+'.nsys-rep'))],check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',action='store_true');p.add_argument('--indices');p.add_argument('--dynamic',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    indices=list(map(int,a.indices.split(','))) if a.indices else read(ROOT/'replay_plan.json')['case_indices']
    try:
        if a.dynamic:
            status=read(ROOT/'state/dynamic_probe_status.json')
            for C in [1,4,8,16]:
                for arm in ARMS:
                    if arm in status and status[arm]['status']=='BLOCKED':continue
                    options=['--depth-replay',str(ROOT/f'dynamic_replay_C{C}.json'),'--repeats','5']
                    if arm in ['T4','TD']:
                        if arm=='TD':options+=['--dynamic']
                        module='tp_runtime'
                    else:module='request_runtime';options+=['--arm',arm]
                    if arm in ['S4','SD']:restore([5,6,7])
                    run(f'depth_replay_{arm}_C{C}_v1',module,ARMS[arm]['GPUs'],options,distributed=arm in ['T4','TD'],timeout=1200)
        elif a.trace:
            profiles=ROOT/'profiles';profiles.mkdir(exist_ok=True)
            status=read(ROOT/'state/dynamic_probe_status.json')
            for arm in ['S4','L4','O4','T4']:
                label='profile_'+arm+'_v1';module='tp_runtime' if arm=='T4' else 'request_runtime'
                if arm=='S4':restore([5,6,7])
                prefix=[PYTHON,'-u']
                if arm=='T4':prefix+=['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4']
                options=['--replay-case','1','--repeats','0','--nsys']+([] if arm=='T4' else ['--arm',arm,'--trace'])
                argv=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node',
                    '--capture-range=cudaProfilerApi','--capture-range-end=stop','--kill=none','-o',str(profiles/label)]+prefix+[
                    '-m','poc_flashvep.ouro_8arm.'+module,'--label',label]+options
                capture_profile(label,argv,ARMS[arm]['GPUs'],a.resume)
            # The short replay and serving runs have different full-capacity KV shapes.
            # Keep the earlier stratum. Add a separately labeled serving-layout stratum.
            for arm in ARMS:
                if arm in status and status[arm]['status']=='BLOCKED':continue
                if arm in ['S4','SD']:restore([5,6,7])
                module='tp_runtime' if arm in ['T4','TD'] else 'request_runtime'
                options=['--limit','4','--cap','32','--serving-KV-shape']
                options+=(['--dynamic'] if arm=='TD' else []) if arm in ['T4','TD'] else ['--arm',arm]
                run('observer_serving_shape_'+arm+'_v1',module,ARMS[arm]['GPUs'],options,distributed=arm in ['T4','TD'],timeout=1200)
                label='profile_serving_shape_'+arm+'_v1';prefix=[PYTHON,'-u']
                if arm in ['T4','TD']:prefix+=['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4']
                argv=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node',
                    '--capture-range=cudaProfilerApi','--capture-range-end=stop','--kill=none','-o',str(profiles/label)]+prefix+[
                    '-m','poc_flashvep.ouro_8arm.'+module,'--label',label]+options+['--nsys']+([] if arm in ['T4','TD'] else ['--trace'])
                capture_profile(label,argv,ARMS[arm]['GPUs'],a.resume)
            for tp in [1,4]:
                label=f'profile_native_TP{tp}_v1'
                if tp==1:restore([5,6,7])
                run(f'observer_control_native_TP{tp}_v1','native_quality',[4] if tp==1 else [4,5,6,7],
                    ['--tp',str(tp),'--limit','4','--cap','32'],native=True,timeout=1200)
                argv=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node',
                    '--capture-range=cudaProfilerApi','--capture-range-end=stop','--kill=none','-o',str(profiles/label),VLLM_PYTHON,'-u',
                    '-m','poc_flashvep.ouro_8arm.native_quality','--tp',str(tp),'--label',label,'--limit','4','--cap','32','--nsys']
                capture_profile(label,argv,[4] if tp==1 else [4,5,6,7],a.resume)
            status=read(ROOT/'state/dynamic_probe_status.json')
            for arm in ['SD','LD','OD','TD']:
                if status[arm]['status']=='BLOCKED':continue
                if arm=='SD':restore([5,6,7])
                module='tp_runtime' if arm=='TD' else 'request_runtime'
                options=['--limit','4','--cap','32']+(['--dynamic'] if arm=='TD' else ['--arm',arm])
                run('observer_control_'+arm+'_v1',module,ARMS[arm]['GPUs'],options,distributed=arm=='TD',timeout=1200)
                label='profile_live_'+arm+'_v1';prefix=[PYTHON,'-u']
                if arm=='TD':prefix+=['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4']
                argv=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node',
                    '--capture-range=cudaProfilerApi','--capture-range-end=stop','--kill=none','-o',str(profiles/label)]+prefix+[
                    '-m','poc_flashvep.ouro_8arm.'+module,'--label',label]+options+['--nsys']+([] if arm=='TD' else ['--trace'])
                capture_profile(label,argv,ARMS[arm]['GPUs'],a.resume)
        else:
            for i,index in enumerate(indices):
                arms=['S4','T4','L4','O4'];arms=arms[i%4:]+arms[:i%4]
                for arm in arms:
                    if arm=='S4':restore([5,6,7])
                    options=['--replay-case',str(index),'--repeats','5']
                    if arm!='T4':options+=['--arm',arm]
                    run(f'replay_{arm}_case{index}_v1','tp_runtime' if arm=='T4' else 'request_runtime',ARMS[arm]['GPUs'],options,distributed=arm=='T4',timeout=1200)
    finally:restore([4,5,6,7])


if __name__=='__main__':main()
