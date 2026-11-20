"""Build a local correlated collector. Do not modify the installed runtime."""
import json
import shutil
import subprocess
import time
from .common import ROOT,SOURCE,PREVIOUS,write,digest,initialize


def main():
    initialize()
    old=PREVIOUS/'references/CUPTI_collector'
    copied=ROOT/'references/CUPTI_collector'
    if not copied.exists():shutil.copytree(old,copied)
    base=json.loads((old/'activity_build.json').read_text())
    folder=ROOT/'references/correlated_collector';folder.mkdir(exist_ok=True)
    source=SOURCE/'correlated_activity.cc';library=folder/'correlated_activity.so'
    previous=folder/'build.json'
    if previous.exists():
        prior=json.loads(previous.read_text())
        write(folder/('build_'+prior['source_SHA256'][:12]+'.json'),prior)
        if library.exists():shutil.copyfile(library,folder/('collector_'+prior['library_SHA256'][:12]+'.so'))
    command=list(base['command'])
    command[command.index(base['source'])]=str(source)
    command[command.index('-o')+1]=str(library)
    subprocess.run(command,check=True,capture_output=True,text=True)
    write(folder/'build.json',dict(time=time.time(),command=command,source=str(source),
        source_SHA256=digest(source),library=str(library),library_SHA256=digest(library),
        CUPTI_library=base['CUPTI_library'],CUPTI_library_SHA256=base['CUPTI_library_SHA256'],
        base_source_SHA256=digest(old/'preallocated_activity.cc'),
        scope='Only external-correlation records and push/pop APIs were added to the prior preallocated collector. This build does not qualify GPU timing.'))
    print('CORRELATED_COLLECTOR_BUILT',digest(library),flush=True)


if __name__=='__main__':main()
