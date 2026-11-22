"""Add strong pure-Token work levels at every nominal interior hybrid frontier."""
import copy
import json
import time
from .common import ROOT,write
from .runner import save,signature
from .work_contract import regenerate,validate

def main():
    baseplan=json.loads((ROOT/'configs/c3_stock_geometry_repair_v6.json').read_text())
    base=next(r for r in baseplan['items'] if r['label']=='StockToken4');items=[]
    for lane,effective in enumerate(((25,4),(23,2))):
        label=f'StockTokenEff{effective[0]}d{effective[1]}';pair=[2*lane,2*lane+1];item=copy.deepcopy(base)
        teacher_cfg=json.loads((ROOT/'configs'/(base['teacher']+'.json')).read_text())
        strict_cfg=json.loads((ROOT/'configs'/(base['strict']+'.json')).read_text())
        strict=f'V2_C3Q_{label}_Strict8_V7';strict_cfg.update(name=strict,gpus=pair,fractional_token_work=list(effective))
        rows=json.loads((ROOT/'data'/strict_cfg['dataset']).read_text())[:strict_cfg['count']]
        regenerate(strict_cfg,rows);validate(strict_cfg,rows);save(strict_cfg)
        teacher=f'V2_C3Q_{label}_OriginalTeacher8_V7'
        teacher_cfg.update(name=teacher,gpus=pair,suffix_by_request=strict_cfg['suffix_by_request']);save(teacher_cfg)
        jobs=[]
        for old_name in base['jobs']:
            cfg=json.loads((ROOT/'configs'/(old_name+'.json')).read_text());name=f'V2_C3Q_{label}_Natural{cfg["count"]}_V7'
            cfg.update(name=name,gpus=pair,fractional_token_work=list(effective))
            regenerate(cfg,json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]);save(cfg);jobs.append(name)
        item.update(label=label,lane=lane,pair=pair,teacher=teacher,strict=strict,jobs=jobs,
            fractional_token_work=list(effective),gate=f'c3_composition_{label}_v7.json',signature=signature(cfg));items.append(item)
    write(ROOT/'configs/c3_fractional_stock_frontier_v7.json',dict(baseplan,time=time.time(),items=items,state_prefix='c3_fractional_stock_frontier_v7',
        scope='Freeze before C3 geometry timings. Add compiled pure-Token effective cuts6.25 and11.5. '
            'They match nominal Hybrid2/f1/8 and Hybrid8/f1/8 work. Token8 andToken12 already cover Hybrid4/f1/8 andHybrid4/f1/4. '
            'Adapter cut4 is only a template. The rational effective cut and explicit suffixes define work. '
            'Report all16-token lattice residuals. Keep original model,compiler,credit0 and common D-cap2. '
            'N8 strict and Natural32/400 state and quality gates remain mandatory.'))
    print('STRONG_PURE_TOKEN_INTERIOR_FRONTIER_FROZEN',flush=True)

if __name__=='__main__':main()
