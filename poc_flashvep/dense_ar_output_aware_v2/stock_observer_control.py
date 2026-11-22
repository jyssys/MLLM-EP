"""Separate request-count/warm-shape changes from numerical observation."""
import json
import time
from .common import ROOT,write,digest
from .runner import save

def prepare():
    source='v2_preemption_observer_fix_v5.zip';sha=digest(ROOT/'references/source_capsules'/source)
    jobs=[]
    for key in ('teacher','strict'):
        prior=f'V2_StockFA3Probe_Token2_{key}_N1_V8'
        cfg=json.loads((ROOT/'configs'/(prior+'.json')).read_text())
        name=f'V2_StockNumericalControl_Token2_{key}_N1_NoAttentionProbe_V9'
        cfg.update(name=name,source_capsule=source,source_capsule_SHA256=sha)
        cfg.pop('stock_attention_probe',None)
        if key=='strict':cfg.update(numerical_probe=False,numerical_probe_sync=False)
        save(cfg);jobs.append(name)
    prior='V2_C3Q_StockToken2_Strict8_CoordinatorFix1_GeometryFixV6'
    cfg=json.loads((ROOT/'configs'/(prior+'.json')).read_text())
    name='V2_StockNumericalControl_Token2_Strict8_WithNumericalProbe_V9'
    cfg.update(name=name,gpus=[0,1],numerical_probe=True,numerical_probe_sync=True)
    save(cfg);jobs.append(name)
    write(ROOT/'jobs/stock_observer_controls_v9.json',dict(time=time.time(),lanes=[jobs,[],[],[]],stop_on_failure=False,
        scope='Diagnostic controls retain source5 and original model kernels. '
            'N1 removes the FA3 snapshot and additional actual-side numerical probe while retaining the same q0 prefix/suffix. '
            'N8 changes only the actual-side numerical observer on the failing configuration,apart from the recorded pair. '
            'N1 also changes maximum/minimum warm shapes. Do not attribute an N1/N8 difference solely to observation. '
            'These runs do not qualify a performance optimization or change a numerical tolerance.'))
    print('STOCK_OBSERVER_CONTROLS_FROZEN',jobs,flush=True)

if __name__=='__main__':prepare()
