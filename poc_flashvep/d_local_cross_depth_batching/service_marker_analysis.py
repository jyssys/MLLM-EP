"""Qualify first-entry markers separately from kernel-activity instrumentation."""
from .common import *
from .analyze import pct,requests,table


def main():
    results=[]
    for model in ('ouro','huginn'):
        reference=ROOT/'raw'/f'local_{model}_native_n128_s2026110701_off'
        base=requests(reference);base_window=json.loads((reference/'window.json').read_text())
        for variant in ('service_once128','service_pool128'):
            name=f'local_{model}_native_{variant}';folder=ROOT/'raw'/name
            receipt=folder/'dispatcher_result.json'
            if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):continue
            data=requests(folder);window=json.loads((folder/'window.json').read_text())
            def summary(rows,origin,left,right):
                itls=[];tpots=[]
                for r in rows:
                    if r['first_token']-origin<=right and r['finish']-origin>=left:tpots.append(r['tpot_ms'])
                    for e in r['token_events']:
                        if not e['first'] and e['new_tokens']==1 and e['previous_time']-origin<=right and e['time']-origin>=left:
                            itls.append(1000*(e['time']-e['previous_time']))
                return [pct(tpots,95),pct(itls,95),pct(itls,99),max(itls,default=None)]
            span=base_window['last_arrival']-base_window['start'];ratios=[]
            for left,right in ((0,span+30),(.1*span,.4*span),(.4*span,.7*span),(.7*span,span+5)):
                a=summary(base,base_window['start'],left,right);b=summary(data,window['start'],left,right)
                ratios.extend(y/x for x,y in zip(a,b) if x is not None and y is not None and x>0)
            duration=(window['end']-window['start'])/(base_window['end']-base_window['start'])
            qualified=bool(ratios and min(ratios)>=.95 and max(ratios)<=1.05 and .95<=duration<=1.05)
            result=dict(name=name,model=model,qualified=qualified,min_local_ratio=min(ratios),max_local_ratio=max(ratios),
                cohort_duration_ratio=duration,
                scope='Full cohort and three fixed arrival-time windows. Compare request TPOT p95, ITL p95/p99, and maximum ITL in both directions. This gate does not provide kernel activity.')
            write(folder/'service_marker_qualification.json',result);results.append(result)
    table(ROOT/'service_marker_qualification.csv',results)
    print(json.dumps(results))


if __name__=='__main__':main()
