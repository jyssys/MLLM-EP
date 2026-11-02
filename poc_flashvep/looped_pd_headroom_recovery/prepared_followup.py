"""Retain prepared-D controls without repeating superseded cold DP curves."""
from .common import *
from .followup_campaign import plan
from .search_plan import make
from .initial_campaign import run
from .primary_campaign import subset_parity


def main():
    ledger=[]
    primary=make()
    for item in plan()['models']:
        reference=next(r['gate'][0] for r in primary['primaries'] if r['model']==item['model'])
        for name in item['prepared_D']:
            write(ROOT/'state/prepared_followup.json',dict(stage='PREPARED_D_DIAGNOSTIC',name=name,time=time.time()))
            okay=run(name) and subset_parity(reference,name)
            ledger.append(dict(name=name,passed=okay))
            write(ROOT/'state/prepared_followup_results.json',ledger)
            if not okay:break
    write(ROOT/'state/prepared_followup.json',dict(stage='PREPARED_D_COMPLETE_REVIEW_REQUIRED',time=time.time(),results=ledger,
        scope='Unstarted cold-graph DP2 and cold profiles are superseded by matched warm-epoch comparisons. Completed old controls remain immutable. Prepared-D results are finite diagnostics, not online capacity.'))
    restore_if_idle()


if __name__=='__main__':main()
