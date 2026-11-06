"""Build the existing activity-only collector without changing earlier tools."""
import subprocess
from .common import *


def prepare():
    initialize()
    from ..fluidtoken_loop_split import common as inherited
    source = REPO / 'poc_flashvep/fluidtoken_loop_split/kernel_activity.cc'
    tools = ROOT / 'tools'
    tools.mkdir(exist_ok=True)
    library = tools / ('kernel_activity_' + digest(source)[:16] + '.so')
    command = ['g++','-O2','-std=c++17','-shared','-fPIC','-pthread',
        '-I/usr/local/cuda-12.8/targets/x86_64-linux/include', str(source),
        '-L/usr/local/cuda-12.8/targets/x86_64-linux/lib',
        '-Wl,-rpath,/usr/local/cuda-12.8/targets/x86_64-linux/lib','-lcupti','-o',str(library)]
    if not library.exists():
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        write(tools / 'build_log.json', dict(stdout=result.stdout, stderr=result.stderr))
    write(tools / 'activity_build.json', dict(command=command, source_SHA256=digest(source),
        library=str(library), library_SHA256=digest(library),
        scope='Reused activity-only collector. GPU coverage and local-tail interference require separate qualification.'))


if __name__ == '__main__':
    prepare()
