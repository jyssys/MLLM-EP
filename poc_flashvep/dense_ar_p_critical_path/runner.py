"""Run immutable jobs through audited resource and correctness gates."""
import sys
from .common import bind

if __name__=='__main__':
    import json
    from .common import ROOT
    runner=bind();plan=json.loads((ROOT/'jobs'/sys.argv[1]).read_text())
    names=plan['lanes'][int(sys.argv[2])]
    if any(json.loads((ROOT/'configs'/(name+'.json')).read_text()).get('stable_worker_source_path') for name in names):
        from .stable_source import install
        runner=install()
    runner.lane(sys.argv[1],int(sys.argv[2]))
