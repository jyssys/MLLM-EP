"""Reject stale request-local partitions before they can enter performance tables."""
import json
from fractions import Fraction
from .common import ROOT
from ..dense_ar_longbench400_pd.geometry import matched_suffixes

def regenerate(cfg,rows):
    if cfg.get('axis_kind')=='Token':
        effective=Fraction(*cfg['fractional_token_work']) if cfg.get('fractional_token_work') else cfg['axis_cut']
        suffix,audit=matched_suffixes([r['input_tokens'] for r in rows],effective)
        audit={k:float(v) if isinstance(v,Fraction) else v for k,v in audit.items()}
        cfg.update(suffix_by_request={f'q{i}':s for i,s in enumerate(suffix)},aggregate_work_audit=audit)
    return cfg

def validate(cfg,rows):
    if cfg.get('axis_kind')!='Token':return True
    table=cfg.get('suffix_by_request',{});n=cfg['count'];rows=rows[:n]
    assert set(table)=={f'q{i}' for i in range(n)},'Stale request-local suffix keys'
    assert all(0<table[f'q{i}']<r['input_tokens'] and table[f'q{i}']%16==0 for i,r in enumerate(rows)), 'Invalid token suffix'
    effective=Fraction(*cfg['fractional_token_work']) if cfg.get('fractional_token_work') else cfg['axis_cut']
    _,expected=matched_suffixes([r['input_tokens'] for r in rows],effective)
    expected={k:float(v) if isinstance(v,Fraction) else v for k,v in expected.items()}
    actual=36*sum(table.values());target=sum(r['input_tokens'] for r in rows)*effective
    assert actual==expected['actual_D_rows'],('Stale or non-matched suffix table',actual,expected['actual_D_rows'])
    assert cfg.get('aggregate_work_audit')==expected,'Missing or stale work audit'
    assert expected['signed_error_rows']==actual-target
    return True

def main():
    cfg=json.loads((ROOT/'configs/V2_C3Q_StockToken8_Natural400.json').read_text())
    validate(cfg,json.loads((ROOT/'data'/cfg['dataset']).read_text()))

if __name__=='__main__':main()
