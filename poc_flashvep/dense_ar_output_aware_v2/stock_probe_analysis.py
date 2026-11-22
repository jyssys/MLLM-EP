"""Locate the first numerical difference at the original FA3 boundary."""
import json
import time
import torch
from .common import ROOT,write,digest
from .c0_analysis import csv_save

def difference(left,right):
    assert left.shape==right.shape,(left.shape,right.shape)
    a=left.double();b=right.double();delta=a-b
    return dict(exact=torch.equal(left,right),shape=list(left.shape),
        max_abs=float(delta.abs().max()),RMS=float(delta.square().mean().sqrt()),
        relative_RMS=float(delta.square().mean().sqrt()/a.square().mean().sqrt().clamp_min(1e-30)),
        differing_elements=int(torch.count_nonzero(delta)),elements=left.numel())

def sampled_attention_error(state):
    # This is a CPU FP64 arithmetic check. It is not a serving implementation.
    count=state['query_count'];start=state['query_start'];heads=state['num_heads'];dim=state['head_size']
    q=state['query'].reshape(count,heads,dim).double()
    kv=state['packed_valid_KV'].double();out=state['attention_output'].reshape(count,heads,dim).double()
    assert kv.shape==(start+count,state['num_kv_heads'],2*dim)
    assert heads%state['num_kv_heads']==0
    rows=[]
    for index in sorted({0,min(127,count-1),count-1}):
        valid=start+index+1
        for head in sorted({0,min(7,heads-1),heads//2,heads-1}):
            key_head=head//(heads//state['num_kv_heads'])
            k=kv[:valid,key_head,:dim];v=kv[:valid,key_head,dim:]
            scores=(k@q[index,head])*state['scale']
            expected=torch.softmax(scores,dim=0)@v
            rows.append(dict(query_relative_index=index,absolute_position=start+index,head=head,
                valid_keys=valid,**difference(expected,out[index,head]),
                evidence='DIAGNOSTIC_CPU_FP64_SAMPLED_ATTENTION',
                scope='Original saved Q/K/V. Exact causal prefix. No model tensor is replaced. No GPU timing claim.'))
    return rows

def analyze():
    torch.set_num_threads(4)
    names={key:f'V2_StockFA3Probe_Token2_{key}_N1_V8' for key in ('teacher','strict')}
    states={}
    for key,name in names.items():
        folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
        if not receipt.exists():print('PROBE_PENDING',name,flush=True);return
        assert json.loads(receipt.read_text())['complete'],name
        states[key]={}
        for path in folder.glob('FA3_layer0_*.pt'):
            value=torch.load(path,map_location='cpu',weights_only=True)
            states[key][value['query_start'],value['query_count']]=(path,value)
        assert states[key],f'No opaque attention snapshots: {name}'
    assert set(states['teacher'])==set(states['strict'])
    rows=[];oracle=[]
    for boundary in sorted(states['teacher']):
        teacher_path,teacher=states['teacher'][boundary];actual_path,actual=states['strict'][boundary]
        row=dict(query_start=boundary[0],query_count=boundary[1],
            teacher_snapshot_SHA256=digest(teacher_path),actual_snapshot_SHA256=digest(actual_path),
            numerical_evidence='DIAGNOSTIC',timing_scope='CPU copies perturb the original serving timing.',
            comparisons={key:difference(teacher[key],actual[key]) for key in ('query','packed_valid_KV','attention_output')},
            metadata_teacher=teacher['metadata'],metadata_actual=actual['metadata'])
        rows.append(row)
        for role,state in (('teacher',teacher),('actual',actual)):
            oracle += [dict(role=role,query_start=boundary[0],query_count=boundary[1],**r) for r in sampled_attention_error(state)]
    result=dict(time=time.time(),runs=names,boundaries=rows,sampled_CPU_FP64_checks=oracle,
        scope='Read-only snapshots follow the unmodified original FA3 call. '
            'Compare identical query boundaries before attributing a difference to transport or Attention. '
            'The sample checks do not validate every operator or the whole model. '
            'A floating-point difference is not itself proof of an incorrect causal mask or an implementation error.')
    write(ROOT/'analysis/stock_small_query_attention_boundary_v8.json',result)
    csv_save(ROOT/'figures/stock_small_query_attention_boundary_v8.csv',rows)
    csv_save(ROOT/'figures/stock_small_query_attention_FP64_samples_v8.csv',oracle)
    print('ORIGINAL_FA3_BOUNDARY_COMPARISONS',[(r['query_start'],r['query_count'],
        {k:(v['exact'],v['max_abs'],v['relative_RMS']) for k,v in r['comparisons'].items()}) for r in rows],flush=True)

if __name__=='__main__':analyze()
