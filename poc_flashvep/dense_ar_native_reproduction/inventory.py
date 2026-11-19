"""Retain unsuccessful runs separately from numerical and observer failures."""
import collections
import json
import time
from .common import ROOT,write


def main():
    lease=json.loads((ROOT/'state/GPU_lease_current.json').read_text());rows=[]
    for file in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(file.read_text())
        if 'entry_module' not in cfg:continue
        folder=ROOT/'raw'/cfg['name'];receipt=folder/'dispatcher_result.json';status=folder/'status.json'
        result=json.loads(receipt.read_text()) if receipt.exists() else {}
        worker=json.loads(status.read_text()) if status.exists() else {}
        state='COMPLETE' if result.get('complete') else ('FAILED' if receipt.exists() or worker.get('status')=='FAILED' else
            ('RUNNING_OR_DRAINING' if lease.get('name')==cfg['name'] and lease.get('status')=='RUNNING' else
             ('INCOMPLETE_PRESERVED' if folder.exists() else 'NOT_RUN')))
        gate=ROOT/'analysis'/(cfg['name']+'_output_gate.json')
        rows.append(dict(name=cfg['name'],status=state,configuration=str(file.relative_to(ROOT)),
            dispatcher=result,worker_status=worker,output_gate=json.loads(gate.read_text()).get('passed') if gate.exists() else None))
    numerical=ROOT/'analysis/standard_Native_Agg_N8_qualification_v1.json'
    numerical_results={k:v['passed'] for k,v in json.loads(numerical.read_text())['results'].items()} if numerical.exists() else {}
    observer={p.name:json.loads(p.read_text())['observer_gate_passed'] for p in (ROOT/'analysis').glob('Native_online_kernel_activity_q*_v1.json')}
    teardown=ROOT/'analysis/UCX_protocol_selection_and_teardown_audit.json'
    teardown_rows=json.loads(teardown.read_text())['post_run_context_errors'] if teardown.exists() else []
    write(ROOT/'analysis/run_and_failure_inventory.json',dict(time=time.time(),runs=rows,
        counts=dict(collections.Counter(r['status'] for r in rows)),BI0_N8_qualification=numerical_results,
        online_observer_gates=observer,post_run_SDK_context_errors=teardown_rows,
        scope='Execution failure, strict numerical non-equivalence, observer failure and post-run SDK errors are different outcomes. COMPLETE does not mean error-free SDK destruction. Preserve every raw folder. Active jobs are not failed jobs.'))
    print(json.dumps(dict(counts=dict(collections.Counter(r['status'] for r in rows)),observer=observer)))


if __name__=='__main__':main()
