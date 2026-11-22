"""Requalify the strong stock model with current-prompt suffix ownership."""
import copy
import json
import time
from .common import ROOT,write,digest
from .runner import save,signature
from .work_contract import regenerate,validate

def main():
    old=json.loads((ROOT/'configs/c3_family_qualification_v4_schedule_fix1.json').read_text())
    source='v2_preemption_observer_fix_v5.zip';sha=digest(ROOT/'references/source_capsules'/source);items=[]
    for previous in old['items']:
        if previous['arm']!='StockToken':continue
        item=copy.deepcopy(previous);teacher_cfg=json.loads((ROOT/'configs'/(item['teacher']+'.json')).read_text())
        strict_cfg=json.loads((ROOT/'configs'/(item['strict']+'.json')).read_text())
        strict=item['strict']+'_GeometryFixV6';strict_cfg.update(name=strict,source_capsule=source,source_capsule_SHA256=sha)
        data=json.loads((ROOT/'data'/strict_cfg['dataset']).read_text())[:strict_cfg['count']]
        regenerate(strict_cfg,data);validate(strict_cfg,data);save(strict_cfg)
        teacher=item['teacher']+'_GeometryFixV6';teacher_cfg.update(name=teacher,source_capsule=source,source_capsule_SHA256=sha,
            suffix_by_request=strict_cfg['suffix_by_request'])
        save(teacher_cfg);jobs=[]
        for previous_name in item['jobs']:
            cfg=json.loads((ROOT/'configs'/(previous_name+'.json')).read_text());name=previous_name+'_GeometryFixV6'
            cfg.update(name=name,source_capsule=source,source_capsule_SHA256=sha)
            regenerate(cfg,json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]);validate(cfg,json.loads((ROOT/'data'/cfg['dataset']).read_text()))
            save(cfg);jobs.append(name)
        item.update(teacher=teacher,strict=strict,jobs=jobs,gate=f'c3_composition_{item["label"]}_geometry_v6.json',signature=signature(cfg));items.append(item)
    write(ROOT/'configs/c3_stock_geometry_repair_v6.json',dict(time=time.time(),items=items,state_prefix='c3_stock_geometry_repair_v6',
        source_capsule=source,source_capsule_SHA256=sha,
        scope='Regenerate token suffixes from the actual original-prompt dataset,request order and cut. '
            'Match aggregate GEMM rows to the nearest legal16-token lattice. Report nonzero residuals. '
            'The strict independent full-model teacher uses exactly the same legal prefix boundary. '
            'Keep default compiled model,continuous scheduler,admission,credits,precision and sampling unchanged. '
            'All earlier stale-map configurations remain invalid and preserved. They cannot enter the primary Stock Token baseline.'))
    print('STRONG_STOCK_CURRENT_DATASET_WORK_CONTRACT_FROZEN',len(items),flush=True)

if __name__=='__main__':main()
