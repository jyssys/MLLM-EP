"""Quantify BI0 geometry differences without changing emitted tokens."""
import json
import time
from .common import ROOT,write,save
from .qualification import configuration
from .run import execute


def difference(actual,reference):
    import torch
    torch.set_num_threads(1)
    a,b=actual.float(),reference.float();delta=(a-b).abs()
    limit=.02+.01*b.abs()
    return dict(elements=a.numel(),max_absolute=float(delta.max()),RMSE=float(delta.square().mean().sqrt()),
        relative_L2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-12)),
        exact=bool(torch.equal(actual,reference)),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        tolerance_exceedances=int((delta>limit).sum()),atol=.02,rtol=.01)


def main():
    names={label:f'Numerical_{label}_ShortAnswer_N8_BI0_v1' for label in ('Standard','Native','Agg')}
    for label,name in names.items():
        extra=dict(reference_single=True,gpus=[1]) if label=='Standard' else {}
        if label=='Agg':extra.update(serial=False,arrival_offsets=[float(i)/2.35 for i in range(8)])
        cfg=configuration(name,'Agg' if label=='Agg' else 'Native','qualification8.json',
            state_capture=False,numerical_probe=True,correctness_only=True,**extra)
        assert execute(save(cfg)),name
    import torch
    rows=[]
    for index in range(8):
        rid=f'q{index}'
        reference=torch.load(ROOT/'raw'/names['Standard']/f'numerical_single_{rid}.pt',weights_only=True)
        for label,role in (('Native_P','prefill'),('Native_D','decode'),('Agg','replica'+str(index%2))):
            folder=ROOT/'raw'/names['Agg' if label=='Agg' else 'Native']
            actual=torch.load(folder/f'numerical_{role}_{rid}.pt',weights_only=True)
            assert torch.equal(actual['positions'],reference['positions'])
            rows.append(dict(request_id=rid,arm=label,length=actual['length'],
                query_start=actual['query_start'],query_count=actual['query_count'],
                final_prompt_hidden=difference(actual['hidden_last'],reference['hidden_last']),
                sampled_KV=difference(actual['KV_samples'],reference['KV_samples'])))
    write(ROOT/'analysis/BI0_cross_geometry_numerical_probe.json',dict(rows=rows,evidence='CORRECTNESS_ONLY',
        scope='32 valid positions per layer and the final prompt hidden. This is not full-KV equality. '
        'Exact Native imported fullKV is a separate gate. Cross-geometry token and task scores remain separate.'))
    write(ROOT/'state/numerical_probe_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/numerical_probe_hold.json',dict(time=time.time(),error=repr(error)));raise
