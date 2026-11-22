"""Draw only measured points and label each evidence scope explicitly."""
import collections
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,write,digest
from .metrics import summarize


def save(fig,name,caption,metadata,manual_layout=False):
    fig.text(.02,.015,caption,fontsize=8,va='bottom')
    if not manual_layout:
        footer_height_inches=.18+.17*len(caption.splitlines())
        bottom=max(.06,footer_height_inches/fig.get_size_inches()[1])
        fig.tight_layout(rect=(0,bottom,1,1))
    paths=[]
    for extension in ('png','pdf'):
        path=ROOT/'figures'/(name+'.'+extension)
        fig.savefig(path,dpi=170,bbox_inches='tight');paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    write(ROOT/'figures'/(name+'_manifest.json'),dict(files=paths,caption=caption,**metadata))


def dataset():
    data=json.loads((ROOT/'data/longbench_balanced400_v1.json').read_text())
    references=json.loads((ROOT/'data/natural_reference_all_candidates.json').read_text())['rows']
    natural={r['prompt_id']:r['output_tokens'] for r in references}
    lengths=[r['input_tokens'] for r in data];outputs=[natural[r['prompt_id']] for r in data]
    meta=dict(evidence='MEASURED_NATURAL_OUTPUT_REFERENCE',n=400,
        dataset_SHA256=digest(ROOT/'data/longbench_balanced400_v1.json'),
        natural_contract='BI0 C1, original BF16,greedy,thinking disabled,EOS allowed,cap2048.')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].hist(np.array(lengths)/1000,bins=30,color='#3465a4',edgecolor='white')
    axes[0].set(xlabel='Input tokens (thousands)',ylabel='Requests',title='Original prompts after Qwen chat template')
    axes[1].hist(outputs,bins=np.arange(0,105,5),color='#2ca58d',edgecolor='white')
    axes[1].set(xlabel='Natural generated output tokens',ylabel='Requests',title='EOS generation, not reference-answer length')
    fig.suptitle('LongBench-Balanced400-v1 | Observed length distributions')
    save(fig,'dataset400_length_distribution','MEASURED NATURAL REFERENCE | N400 | 1 generation/request | Selected output1–99; not production traffic.',meta)
    fig,ax=plt.subplots(figsize=(10,5))
    for task in sorted({r['task'] for r in data}):
        rows=[r for r in data if r['task']==task]
        ax.scatter([r['input_tokens']/1000 for r in rows],[natural[r['prompt_id']] for r in rows],s=18,alpha=.7,label=f'{task} (N{len(rows)})')
    ax.set(xlabel='Input tokens (thousands)',ylabel='Natural generated output tokens',title='LongBench400 | Input × output by task')
    ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8);ax.grid(alpha=.2)
    save(fig,'dataset400_input_output_scatter','MEASURED NATURAL REFERENCE | N400 | No filler, duplication or new context truncation | Natural cap2048.',meta)
    freeze=json.loads((ROOT/'data/balanced400_freeze_manifest.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axes,('available_cells','selected_cells'),('Eligible candidates with output 1–99','Frozen selected 400')):
        values=np.array(freeze[key]);picture=ax.imshow(values,cmap='Blues',vmin=0,vmax=values.max())
        for (i,j),value in np.ndenumerate(values):ax.text(j,i,str(value),ha='center',va='center',color='black')
        ax.set_xticks(range(4),['1–8','9–24','25–49','50–99']);ax.set_yticks(range(4),['3K–6K','6K–10K','10K–20K','20K–36K'])
        ax.set(xlabel='Natural output tokens',ylabel='Input tokens',title=title)
        fig.colorbar(picture,ax=ax,shrink=.75,label='Requests')
    save(fig,'dataset400_balance_audit','MEASURED NATURAL REFERENCE | Declared 13-task universe | Exact 25/cell is NOT achieved. Input 100/bin is achieved.',meta)


def motivation():
    jobs=json.loads((ROOT/'jobs/motivation400_discovery_v1.json').read_text())['jobs']
    groups=collections.defaultdict(list);sources=[]
    for name in jobs:
        folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text())['complete']:continue
        row=summarize(name);groups[row['method'],row['offered_QPS']].append(row);sources.append(name)
    if not sources:return
    colors={'Agg':'#d95f02','Native':'#009e73'}
    for field,label,stem in [('TTFT_p95_ms','TTFT p95 (ms)','ttft_p95'),
            ('ITL_p99_ms','Client-visible ITL p99 (ms)','itl_p99'),
            ('TPOT_p95_ms','Request TPOT p95 (ms/token)','tpot_p95'),
            ('generated_output_tokens_per_second','Generated output tokens/s','output_tokens'),
            ('SLO_qualified_output_tokens_per_second','SLO-qualified output tokens/s','slo_goodput')]:
        fig,ax=plt.subplots(figsize=(7.5,4.8))
        for method in ('Agg','Native'):
            points=sorted((qps,rows) for (kind,qps),rows in groups.items() if kind==method)
            if not points:continue
            x=[q for q,_ in points];means=[float(np.mean([r[field] for r in rows])) for _,rows in points]
            ax.plot(x,means,'o-',color=colors[method],label='Agg DP2' if method=='Agg' else 'Native PD 1P1D')
            for (q,rows),mean in zip(points,means):ax.annotate('n='+str(len(rows)),(q,mean),xytext=(3,5),textcoords='offset points',fontsize=7)
        ax.set(xlabel='Offered QPS (requests/s)',ylabel=label,title='LongBench400 | BI0 | 2 × H100 | B8192/S1024')
        ax.grid(alpha=.2);ax.legend()
        save(fig,'motivation400_'+stem,
            'MEASURED ONLINE | Discovery:1 start/point; no start-level CI yet | Natural EOS cap2048 | Finite cohort, not sustainable capacity.\nMissing QPS points are NOT_RUN. Lines connect measured points only. Quality and decode gates are separate.',
            dict(evidence='MEASURED_ONLINE_DISCOVERY',sources=sources,metric=field,
                dataset_SHA256=digest(ROOT/'data/longbench_balanced400_v1.json'),independent_starts_per_point=1))


if __name__=='__main__':
    import sys
    dataset() if len(sys.argv)==1 or sys.argv[1]=='dataset' else motivation()
