"""Verify deferred CUPTI records with eager and graph-launched CUDA work."""
import csv
import os
import subprocess
import sys
from .common import *


def child():
    import torch
    from .full_activity import library,anchor
    torch.cuda.set_device(0)
    a=torch.ones(1024,device='cuda');b=torch.empty_like(a)
    for _ in range(10):torch.add(a,1,out=b)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):torch.add(a,1,out=b)
    torch.cuda.synchronize()
    lib=library(override=os.environ.get('ACTIVITY_SELFTEST_LIBRARY'))
    assert lib.activity_begin()==0;begin=anchor(lib)
    for _ in range(100):torch.add(a,1,out=b);graph.replay()
    torch.cuda.synchronize();end=anchor(lib)
    label=os.environ.get('ACTIVITY_SELFTEST_LABEL','deferred_activity')
    path=ROOT/'tools'/(label+'_selftest.csv')
    code=lib.activity_end(str(path).encode())
    rows=list(csv.DictReader(path.open()));kernels=[r for r in rows if r['kind']=='kernel']
    result=dict(result=code,kernels=len(kernels),dropped=lib.activity_dropped(),invalid=lib.activity_invalid(),
        graph_kernels=sum(int(r['graph_node'])!=0 for r in kernels),begin=begin,end=end,
        source_sha=digest(SOURCE/'deferred_activity.cc'),
        binary_sha=digest(os.environ.get('ACTIVITY_SELFTEST_LIBRARY',ROOT/'tools/deferred_activity.so')))
    result['passed']=code==0 and len(kernels)==200 and result['graph_kernels']==100 and not result['dropped'] and not result['invalid']
    write(ROOT/'tools'/(label+'_selftest.json'),result);print(json.dumps(result))
    assert result['passed']


def main():
    initialize()
    if '--child' in sys.argv:return child()
    from ..looped_pd_autonomous_headroom.common import resources
    owner=resources()
    try:
        owner.stop(list(range(8)))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
        subprocess.run([sys.executable,'-m','poc_flashvep.completion_aware_loop_pd.activity_selftest','--child'],env=env,check=True)
    finally:
        if not owner.processes():owner.restore(list(range(8)))


if __name__=='__main__':main()
