"""Report state changes without changing experiment configuration."""
import json
import time
from .common import ROOT


def snapshot():
    lease=ROOT/'state/GPU_lease_current.json'
    result=json.loads(lease.read_text()) if lease.exists() else {}
    result['completed_runs']=sum(json.loads(p.read_text()).get('complete',False)
        for p in (ROOT/'raw').glob('*/dispatcher_result.json'))
    result['holds']=[p.name for p in (ROOT/'state').glob('*hold.json')]
    result['resolved_holds']=['last12_confirmation_hold.json'] if (ROOT/'state/new_overlay_runtime_smoke_complete.json').exists() else []
    acknowledgement=ROOT/'state/acknowledged_optional_failures.json'
    result['acknowledged_failures']=json.loads(acknowledgement.read_text())['holds'] if acknowledgement.exists() else []
    result['timestamp']=time.strftime('%Y-%m-%d %H:%M:%S %Z')
    closed=ROOT/'state/CAMPAIGN_CLOSED.json'
    if closed.exists():result['campaign_status']='GPU_WORK_CLOSED'
    return result


def main():
    previous=None
    for _ in range(180):
        row=snapshot();key={k:v for k,v in row.items() if k!='timestamp'}
        if key!=previous:print(json.dumps(row),flush=True);previous=key
        time.sleep(10)


if __name__=='__main__':main()
