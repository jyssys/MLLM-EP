import subprocess
from .common import *


def main():
    initialize();source=SOURCE/'activity_deferred.cc'
    target=ROOT/'tools'/('activity_deferred_'+digest(source)[:16]+'.so')
    command=['g++','-O2','-std=c++17','-shared','-fPIC','-pthread',
        '-I/usr/local/cuda-12.8/targets/x86_64-linux/include',str(source),
        '-L/usr/local/cuda-12.8/targets/x86_64-linux/lib',
        '-Wl,-rpath,/usr/local/cuda-12.8/targets/x86_64-linux/lib','-lcupti','-o',str(target)]
    if not target.exists():subprocess.run(command,check=True)
    write(ROOT/'tools/activity_build_deferred.json',dict(command=command,source_SHA256=digest(source),
        warm_source_SHA256=digest(SOURCE/'activity_warm.cc'),
        included_source_SHA256=digest(REPO/'poc_flashvep/fluidtoken_loop_split/kernel_activity.cc'),
        library=str(target),library_SHA256=digest(target)))
    jobs=[]
    for cap in (4,16):
        cfg=json.loads((ROOT/'configs'/f'idle_output{cap}_capture_n64.json').read_text())
        cfg.update(name=f'idle_output{cap}_deferred_n64',idle_no_midrun_disable=True,
            idle_activity_build='activity_build_deferred.json',
            phase='NO_MIDRUN_PROFILER_DISABLE_CONTROL')
        save(cfg);jobs.append(cfg['name'])
    jobs+=json.loads((ROOT/'state/drain_audit_plan.json').read_text())['jobs']
    write(ROOT/'state/output_idle_extra_plan.json',dict(jobs=jobs,
        reason='The first short capture disables activity during Serving. That call lasts 478 ms on P in the Output4 control. Defer disable and export until after outputs. Keep the five-second analysis window.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
