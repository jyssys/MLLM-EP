"""Build a separate kernel/copy-only observer without altering old collectors."""
import json
import subprocess
from .common import ROOT,SOURCE,write,digest


def main():
    base=json.loads((ROOT/'references/correlated_collector_v1/build.json').read_text())
    prior=json.loads((ROOT/'references/correlated_collector_v1/build_result.json').read_text())
    folder=ROOT/'references/kernel_only_collector_v3';folder.mkdir(exist_ok=True)
    source=SOURCE/'activity_kernel_only_v3.cc';library=folder/'activity.so'
    assert not library.exists(),'Use a new version after any source change'
    command=list(prior['command']);command[command.index(base['source'])]=str(source)
    command[command.index('-o')+1]=str(library)
    result=subprocess.run(command,capture_output=True,text=True)
    write(folder/'build_result.json',dict(command=command,stdout=result.stdout,stderr=result.stderr,returncode=result.returncode))
    result.check_returncode()
    write(folder/'build.json',dict(source=str(source),source_SHA256=digest(source),library=str(library),
        library_SHA256=digest(library),CUPTI_library=base['CUPTI_library'],CUPTI_library_SHA256=base['CUPTI_library_SHA256'],
        scope='Concurrent kernel activity and optional copies only. No driver/runtime/external/graph tracing. '
            'No operator attribution. A separate clean/profile/clean observer gate remains required.'))
    print('BUILT_KERNEL_ONLY',digest(library),flush=True)


if __name__=='__main__':main()
