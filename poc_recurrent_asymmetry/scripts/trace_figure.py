"""Plot actual correlated intervals without rescaling them to clean latency."""
import argparse
import gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_recurrent_asymmetry.contract import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    rows=json.load(gzip.open(ROOT/'metrics/nsight/events.json.gz','rt'))
    selected=[r for r in rows if r['request']=='ASYM_REQUEST|P=16384|rep=0|labels=True'
        and r['phase']=='ASYM_PHASE|DECODE|step=0']
    assert selected
    origin=min(r['start'] for r in selected)
    categories=['ATTN_CORE','KV_UPDATE','MLP','QKV','O_PROJ','NORM','OTHER']
    fig,ax=plt.subplots(figsize=(12,4))
    for index,category in enumerate(categories):
        spans=[(r['start']-origin,r['end']-r['start']) for r in selected
            if (r['category'] if r['category'] in categories else 'OTHER')==category]
        ax.broken_barh(spans,(index-.35,.7),facecolors=f'C{index}')
    ax.set(yticks=range(len(categories)),yticklabels=categories,xlabel='Actual GPU clock ms from first kernel',
        title='MEASURED physical GPU4, 16K cached forward. Heavy observer: +13.36%. Not clean latency.')
    fig.tight_layout();fig.savefig(args.output/'actual_trace_timeline.png',dpi=170);plt.close(fig)
    write(args.output/'timeline_contract.json',dict(physical_gpu=4,request=selected[0]['request'],
        phase=selected[0]['phase'],actual_events=len(selected),clock='One Nsight GPU clock',
        envelope_ms=max(r['end'] for r in selected)-origin,
        validity='Heavy instrumented trace only. Empty spans are not proven removable waits.',
        simultaneous_four_rank_alignment='NOT_ESTABLISHED'))


if __name__=='__main__':main()
