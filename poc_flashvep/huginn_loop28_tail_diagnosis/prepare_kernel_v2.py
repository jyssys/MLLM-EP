import subprocess
from .common import *


def main():
    initialize();source=SOURCE/'activity_warm.cc';target=ROOT/'tools'/('activity_'+digest(source)[:16]+'.so')
    target.parent.mkdir(exist_ok=True)
    command=['g++','-O2','-std=c++17','-shared','-fPIC','-pthread',
        '-I/usr/local/cuda-12.8/targets/x86_64-linux/include',str(source),
        '-L/usr/local/cuda-12.8/targets/x86_64-linux/lib',
        '-Wl,-rpath,/usr/local/cuda-12.8/targets/x86_64-linux/lib','-lcupti','-o',str(target)]
    if not target.exists():subprocess.run(command,check=True)
    write(ROOT/'tools/activity_build_v2.json',dict(command=command,source_SHA256=digest(source),
        included_source_SHA256=digest(REPO/'poc_flashvep/fluidtoken_loop_split/kernel_activity.cc'),
        library=str(target),library_SHA256=digest(target),scope='CPU activity collector. Warm initialization and fixed storage reservation occur before the measured cohort.'))
    jobs=[]
    for arm in ('zero','full'):
        cfg=json.loads((ROOT/'configs'/f'tail_{arm}_kernel_tail_n64.json').read_text())
        cfg.update(name=f'tail_{arm}_kernel_v2_n64',diagnosis_kernel_revision2=True)
        save(cfg);jobs.append(cfg['name'])
    write(ROOT/'state/kernel_v2_plan.json',dict(jobs=jobs,
        reason='The first Stream-ID query loads a second CUPTI library and returns status 39. Query through the active collector handle instead. Warm the collector before measurement. Retain the first capture unchanged.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
