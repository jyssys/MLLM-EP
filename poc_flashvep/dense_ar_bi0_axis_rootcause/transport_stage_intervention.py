"""Compare paired host stages for the exact-byte coalescing intervention."""
import itertools
import json
import numpy as np
from .common import ROOT,write
from .historical_bridge_analysis import current


def paired_difference(a,b):
    assert len(a)==len(b)==3
    differences=np.asarray(a)-np.asarray(b)
    draws=[float(differences[list(indices)].mean()) for indices in itertools.product(range(3),repeat=3)]
    return dict(mean_difference_ms=float(differences.mean()),
        CI95=[float(np.percentile(draws,p)) for p in (2.5,97.5)],starts=3,resamples=27)


def main():
    path=ROOT/'analysis/coalescing_paired_repetition_v1.json'
    if not path.exists():return
    original=json.loads(path.read_text())['results']
    rows=[current(r) for r in original if r['status']=='DIAGNOSTIC']
    cells={(r['axis'],r['coalesced'],r['seed']):r for r in rows}
    seeds=(2026111920,2026111921,2026111922)
    if len(cells)!=12:return
    fields=list(rows[0]['mean_stages_ms']);changes={}
    for axis in ('Layer','Token'):
        changes[axis]={field:paired_difference(
            [cells[axis,True,s]['mean_stages_ms'][field] for s in seeds],
            [cells[axis,False,s]['mean_stages_ms'][field] for s in seeds]) for field in fields}
    axis_means={}
    for axis in ('Layer','Token'):
        axis_means[axis]={str(coalesced):{field:float(np.mean([cells[axis,coalesced,s]['mean_stages_ms'][field] for s in seeds]))
            for field in fields} for coalesced in (False,True)}
    write(ROOT/'analysis/coalescing_host_stage_intervention.json',dict(rows=rows,
        coalescing_minus_no_coalescing_host_stages=changes,mean_host_stages_ms=axis_means,
        evidence='DIAGNOSTIC',physical_coalescing_intervention='NOT_REALIZED',
        scope='BI1 coalescing host flag only. Actual descriptor counts stay1764 before/after. No physical merge gain is identified. '
        'Three paired starts include one discovery and two new seeds. '
        'Stage confidence intervals describe host boundaries. They are not exclusive DMA/kernel times. '
        'All factor outcomes, including absolute regressions, remain visible. No matched BI1 Native-relative safety claim.'))
    print(json.dumps(dict(changes=changes),indent=2))


if __name__=='__main__':main()
