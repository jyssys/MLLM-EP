"""Qualify the minimal Token Client alias correction as a new version."""
import json
import time
from .common import ROOT,SOURCE,save,write,digest
from .batch_matched import ARMS
from .fixed_cohort_probe_v2 import configuration as previous_configuration
from .qualification import gate
from .run import execute


def configuration(method,layers):
    cfg=previous_configuration(method,layers)
    cfg.update(name=cfg['name'].removesuffix('_v2')+'_v3',cohort_runtime_version=3,
        entry_module='poc_flashvep.dense_ar_matched_axis_pd.fixed_matched_cohort_v3')
    return cfg


def main():
    configs=[configuration(a,m) for a,m in ARMS]
    metadata=dict(names=[c['name'] for c in configs],phase='CORRECTNESS_ONLY',
        runtime_adapter_hashes={n:digest(SOURCE/n) for n in ('fixed_matched_cohort.py',
            'fixed_matched_cohort_v2.py','fixed_matched_cohort_v3.py')},
        reason='Token PrefixChannel replaces worker.Client from ar_worker.Client. Preserve the registered cohort Client at that exact alias boundary. The local submission frontier and model forward stay unchanged.',
        performance_starts_seen=0)
    path=ROOT/'state/fixed_cohort2_v3_qualification_design.json'
    if path.exists():
        assert {k:v for k,v in json.loads(path.read_text()).items() if k!='epoch'}==metadata
    else:
        write(path,dict(epoch=time.time(),**metadata))
    results=[]
    try:
        for cfg in configs:
            assert execute(save(cfg)),cfg['name']
            results.append(gate(cfg['name'],configs[0]['name']))
            write(ROOT/'state/fixed_cohort2_v3_qualification_progress.json',dict(epoch=time.time(),
                completed=len(results),total=len(configs),last=cfg['name']))
        write(ROOT/'state/fixed_cohort2_v3_qualification_complete.json',dict(epoch=time.time(),
            passed=True,names=[r['name'] for r in results],evidence='CORRECTNESS_ONLY'))
    except BaseException as error:
        write(ROOT/'state/fixed_cohort2_v3_qualification_stopped.json',dict(epoch=time.time(),
            error=repr(error),evidence='FAILED'))
        raise


if __name__=='__main__':
    main()
