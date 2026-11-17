"""Record resumable run progress without changing experiments."""
import collections
import json
import time
from .common import ROOT,write


def main():
    prior=set()
    while not (ROOT/'state/CAMPAIGN_CLOSED.json').exists():
        results={p.parent.name:json.loads(p.read_text()) for p in (ROOT/'raw').glob('*/dispatcher_result.json')}
        complete={name for name,row in results.items() if row.get('complete')}
        new=sorted(complete-prior)
        phase=collections.Counter()
        for name in complete:
            cfg=json.loads((ROOT/'raw'/name/'configuration.json').read_text())
            phase[cfg.get('phase','UNSPECIFIED')]+=1
        freeze=[p.name for p in (ROOT/'state').glob('candidate_confirmation_freeze*')]
        progress=dict(time=time.time(),complete=len(complete),failed=len(results)-len(complete),
            by_phase=dict(phase),candidate_freezes=freeze,
            scope='Dispatcher completion is not a substitute for model-state, output or performance qualification.')
        write(ROOT/'state/monitor_progress.json',progress)
        if new:
            print(json.dumps(dict(complete=len(complete),failed=len(results)-len(complete),
                newly_completed=new[-4:],candidate_freezes=freeze)),flush=True)
        prior=complete;time.sleep(30)
    print('MONITOR_ADMISSION_CLOSED',flush=True)


if __name__=='__main__':
    main()
