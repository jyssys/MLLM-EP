"""Choose fixed cuts from discovery only, before held-out starts."""
import json
import time
from .common import ROOT,write,digest


def main():
    assert (ROOT/'state/mechanism_discovery_complete.json').exists()
    design=json.loads((ROOT/'state/mechanism_discovery_design.json').read_text())
    names=[n for n in design['names'] if '_Native_' not in n]
    values=[]
    for name in names:
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        metric=json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text())
        slo=metric['SLO_regimes']['Common_Balanced_OutputSweep']
        values.append(dict(name=name,axis=cfg['evaluation_arm'],layers=cfg['static_layers'],
            QPS=cfg['arrival_rate'],output=cfg['cap'],batch=cfg['D_admission_batch'],
            seed=cfg['arrival_seed'],qualified_output_tokens_s=slo['Output_token_SLO_goodput'],
            TTFT_p95_ms=metric['TTFT_p95_ms'],raw_output_tokens_s=metric['output_tokens_s'],
            metric_SHA256=digest(ROOT/'analysis'/(name+'_metrics.json'))))
    selections=[]
    for q in (2.,6.):
        candidates=[r for r in values if r['QPS']==q]
        key=lambda r:(r['qualified_output_tokens_s'],-r['TTFT_p95_ms'],r['raw_output_tokens_s'])
        selected={a:max([r for r in candidates if a=='Overall' or r['axis']==a],key=key)
            for a in ('Layer','Token','Overall')}
        selections.append(dict(QPS=q,output=16,batch=1,count=72,selection=selected))
    document=dict(discovery=values,selections=selections,
        selection_order=['Max common-Balanced qualified output token/s','Min TTFT p95','Max raw output token/s'],
        scope='One fresh discovery start per action/QPS. Freeze Last2/4/8/12 selections before new held-out seeds. Native remains an external measured baseline. No best-vs-best global tuning or held-out winner selection is implied. This freeze covers only Output16/Decode1/QPS2 or6.')
    target=ROOT/'configs/frozen_best_fixed_axes_v1.json'
    if target.exists():
        assert json.loads(target.read_text())['selection']==document
    else:
        write(target,dict(frozen_epoch=time.time(),selection=document))
    print('BEST_FIXED_AXES_FROZEN',[(r['QPS'],r['selection']['Overall']['axis'],r['selection']['Overall']['layers'])
        for r in selections],flush=True)


if __name__=='__main__':
    main()
