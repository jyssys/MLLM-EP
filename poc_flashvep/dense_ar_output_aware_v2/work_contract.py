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

def actual_partitions(cfg,folder,rows):
    from ..dense_ar_longbench400_pd.geometry import Partition,aligned_suffix
    kind=cfg.get('axis_kind')
    if kind not in ('Layer','Token','Hybrid'):return None
    expected={}
    for i,row in enumerate(rows[:cfg['count']]):
        rid=f'q{i}';length=row['input_tokens'];upper=cfg['axis_cut'] if kind!='Token' else 0;suffix=0
        if kind=='Token':suffix=cfg['suffix_by_request'][rid]
        elif kind=='Hybrid':
            fraction=cfg['hybrid_suffix_fraction'];suffix=0 if not fraction else min(max(16,int(round(length*fraction/16))*16),(length-1)//16*16)
        if rid in cfg.get('suffix_by_request',{}):suffix=cfg['suffix_by_request'][rid]
        expected[rid]=Partition(length,upper,suffix).record()
    for role in ('prefill','decode'):
        records=json.loads((folder/f'axis_events_{role}.json').read_text())['rows']
        actual={r['request_id']:r for r in records if r['event']=='PARTITION' and r['request_id'].startswith('q')}
        assert set(actual)==set(expected),('Missing actual partition records',cfg['name'],role)
        for rid,wanted in expected.items():
            for key,value in wanted.items():
                if key=='scope':continue
                assert actual[rid][key]==value,('Actual partition differs from the frozen action',cfg['name'],role,rid,key)
    return dict(P_rows=sum(p['P_rows'] for p in expected.values()),D_rows=sum(p['D_rows'] for p in expected.values()),
        P_attention_pairs=sum(p['P_attention_pairs'] for p in expected.values()),D_attention_pairs=sum(p['D_attention_pairs'] for p in expected.values()),
        offloaded_requests=sum(bool(p['upper'] or p['suffix']) for p in expected.values()),fallback_requests=0,
        scope='Actual producer and consumer partition records match every prescribed static action. Query/layer coverage is a separate runtime audit.')

def executed_prompt_work(cfg,folder,rows):
    """Count valid executed prompt rows. Retain legitimate preemption repeats."""
    lengths={f'q{i}':r['input_tokens'] for i,r in enumerate(rows[:cfg['count']])}
    result={}
    for role in ('prefill','decode','replica0','replica1'):
        path=folder/f'actual_queries_{role}.json'
        if not path.exists():continue
        queries=[r for r in json.loads(path.read_text())['rows'] if r['request_id'] in lengths]
        chunks={rid:0 for rid in lengths};prompt_tokens=0;batches={}
        for row in queries:
            rid=row['request_id'];valid=max(0,min(row['start']+row['count'],lengths[rid])-row['start'])
            prompt_tokens+=valid;chunks[rid]+=int(valid>0);batches.setdefault(row['batch_id'],row)
        axis=folder/f'axis_events_{role}.json'
        if cfg.get('axis_kind') and axis.exists():
            visits=[r for r in json.loads(axis.read_text())['rows'] if r.get('event')=='MODEL_VISIT' and r['request_id'] in lengths]
            token_layer_rows=sum(max(0,min(r['start']+r['count'],lengths[r['request_id']])-r['start'])*len(r['layers']) for r in visits)
            observation='Recorded actual model-visit query intervals and layer lists.'
        else:
            token_layer_rows=36*prompt_tokens
            observation='Unmodified full36 model and actual runner query intervals.'
        result[role]=dict(valid_prompt_query_tokens=prompt_tokens,valid_executed_prompt_token_layer_rows=token_layer_rows,
            per_request_prompt_query_chunks=chunks,
            batch_valid_query_rows_including_decode=sum(r['actual_batch_queries'] for r in batches.values()),
            batch_padded_query_rows_including_decode=sum(r['padded_queries'] for r in batches.values()),observation=observation)
    nominal=actual_partitions(cfg,folder,rows) if cfg.get('axis_kind') else None
    minimum=36*sum(lengths.values());total=sum(r['valid_executed_prompt_token_layer_rows'] for r in result.values())
    return dict(roles=result,nominal_static_partition=nominal,ideal_work_conserving_full36_rows=minimum,
        observed_total_valid_prompt_token_layer_rows=total,observed_total_minus_ideal_rows=total-minimum,
        actual_P_minus_nominal_rows=result['prefill']['valid_executed_prompt_token_layer_rows']-nominal['P_rows'] if nominal else None,
        actual_D_minus_nominal_rows=result['decode']['valid_executed_prompt_token_layer_rows']-nominal['D_rows'] if nominal else None,
        scope='Executed valid prompt token-layer rows count all recorded recomputation,including normal preemption. '
            'Padding,autoregressive decode,embedding,LM head,Norm,Attention pairs and transfer are not GEMM-row equivalents. '
            'A static partition record alone does not prove exact executed work. Native proxy last-prompt recomputation remains in its D rows. '
            'Extra rows can represent official proxy recomputation or normal preemption. They do not alone prove an implementation error.')

def main():
    cfg=json.loads((ROOT/'configs/V2_C3Q_StockToken8_Natural400.json').read_text())
    validate(cfg,json.loads((ROOT/'data'/cfg['dataset']).read_text()))

if __name__=='__main__':main()
