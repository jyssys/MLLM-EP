"""Compile the preserved collector with explicit buffer diagnostics."""
import json
import subprocess
from .common import ROOT,REPO,SOURCE,write,digest


def main():
    previous=REPO/'poc_flashvep/reports/dense_ar_bi0_axis_rootcause_20261120/references/correlated_collector/build.json'
    base=json.loads(previous.read_text())
    folder=ROOT/'references/correlated_collector_v1';folder.mkdir(exist_ok=True)
    source=SOURCE/'activity_collector.cc';library=folder/'activity.so'
    assert not library.exists(),'Preserve prior collector binaries'
    command=list(base['command']);command[command.index(base['source'])]=str(source);command[command.index('-o')+1]=str(library)
    result=subprocess.run(command,capture_output=True,text=True)
    write(folder/'build_result.json',dict(command=command,stdout=result.stdout,stderr=result.stderr,returncode=result.returncode))
    result.check_returncode()
    write(folder/'build.json',dict(source=str(source),source_SHA256=digest(source),library=str(library),library_SHA256=digest(library),
        CUPTI_library=base['CUPTI_library'],CUPTI_library_SHA256=base['CUPTI_library_SHA256'],
        predecessor_source_SHA256=base['source_SHA256'],
        scope='The prior callback and collector remain unchanged except exported buffer diagnostics. '
            'The Python controller can enable collection after warmup. This build does not qualify timing.'))
    print('BUILT',digest(library))


if __name__=='__main__':main()
