"""Test BI0 start variation without accepting a failed observer proof."""
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,digest,bind
from .bootstrap import baseline,launch
from .capsule import build


def prepare():
    runner=bind();source=build(json.loads((ROOT/'configs/PC_Metadata_StockToken.json').read_text())['source_capsule'].removeprefix('pc_'))
    names=[]
    for label in ('OriginalRepeat','NewSourceNoEvents'):
        cfg=json.loads((ROOT/'configs/PC_Q_StockToken_Original_Strict8_ArrivalFix2.json').read_text())
        name=f'PC_StockStateDiagnostic_{label}_N8';cfg['name']=name
        if label=='NewSourceNoEvents':
            cfg.update(source_capsule=source.name,source_capsule_SHA256=digest(source),
                P_event_timing=False,P_critical_trace=True,source_credit_release_on_retirement=False)
        runner.save(cfg);names.append(name)
    tests=[]
    for start,seed in enumerate((2026112281,2026112309,2026112310)):
        jobs=[]
        for stage in ('Clean1','Profile','Clean2'):
            name=f'PC_StockOriginalKernelCPC_s{start}_{stage}'
            cfg=baseline('StockToken',name,[6,7],seed=seed,save_cfg=False)
            cfg.update(kernel_activity=stage=='Profile',kernel_activity_roles=['prefill'],
                collector_namespace='kernel_only_collector_v3',runtime_api_activity=False,activity_copies=False,
                operator_correlation=False,graph_enclosures=False,activity_delay_seconds=0,
                activity_ROI_seconds=None,kernel_activity_seconds=86400,
                observer_scope='Original compiled Stock Token source and model-runner method. Kernel-only full trace. '
                    'No event-method clone or source-retirement code. Require unchanged5% CPC timing gate. '
                    'Per-request forward event attribution remains NOT_MEASURED.')
            runner.save(cfg);jobs.append(name);names.append(name)
        tests.append(dict(label='StockToken',start=start,seed=seed,pair=[6,7],jobs=jobs))
    path=ROOT/'jobs/stock_original_kernel_cpc_and_state_diagnostics.json';assert not path.exists()
    write(path,dict(time=time.time(),lanes=[[],[],[],names],tests=tests,stop_on_failure=True,
        diagnostic_names=names[:2],
        scope='The original observer strict failure remains FAILED. Two fixed counterfactual controls test start variation and source wrapping. '
            'Run both once. Do not select a passing repeat as evidence that an observer fixed an error. '
            'The following CPC tests use only the original qualified runtime. They do not qualify the rejected event observer.'))
    return launch(path.name)


if __name__=='__main__':print(prepare(),flush=True)
