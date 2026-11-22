"""Preserve v1 and export already-collected API timestamps in a new binary."""
import json
import subprocess
from .common import ROOT,SOURCE,write,digest


def main():
    base=json.loads((ROOT/'references/correlated_collector_v1/build.json').read_text())
    prior=json.loads((ROOT/'references/correlated_collector_v1/build_result.json').read_text())
    folder=ROOT/'references/correlated_collector_api_v2';folder.mkdir(exist_ok=True)
    source=SOURCE/'activity_collector_api_v2.cc';library=folder/'activity.so'
    assert not library.exists(),'Preserve the existing binary and use a new version'
    command=list(prior['command']);command[command.index(base['source'])]=str(source)
    command[command.index('-o')+1]=str(library)
    result=subprocess.run(command,capture_output=True,text=True)
    write(folder/'build_result.json',dict(command=command,stdout=result.stdout,stderr=result.stderr,returncode=result.returncode))
    result.check_returncode()
    write(folder/'build.json',dict(source=str(source),source_SHA256=digest(source),library=str(library),
        library_SHA256=digest(library),CUPTI_library=base['CUPTI_library'],CUPTI_library_SHA256=base['CUPTI_library_SHA256'],
        predecessor_source_SHA256=base['source_SHA256'],
        scope='Same activity collection as v1. Export every recorded CUDA API timestamp instead of graph-only API timestamps. '
            'No GRAPH_TRACE activity is enabled. Export occurs after measurement. A new observer gate remains required.'))
    print('BUILT',digest(library),flush=True)


if __name__=='__main__':main()
