"""Describe frozen inputs and measured Natural EOS outputs without resampling."""
import collections
import json
import sys
import numpy as np
from .common import ROOT,REPO,write,digest


def stats(values):
    values=np.asarray(values,dtype=float)
    return dict(n=len(values),mean=float(values.mean()),min=float(values.min()),max=float(values.max()),
        **{f'p{p}':float(np.percentile(values,p)) for p in (10,25,50,75,90,95,99)})


def main():
    data_path=ROOT/'data/longbench_v1_mixed200.json'
    data=json.loads(data_path.read_text())
    job=sys.argv[1] if len(sys.argv)>1 else 'standard_HND_BI0_LongBench200_natural_C1_cap2048_v1'
    suffix='_BI1' if '_BI1_' in job else ''
    reference=ROOT/'raw'/job/'natural_reference.json'
    generated={r['prompt_id']:r for r in json.loads(reference.read_text())['rows']}
    assert len(generated)==len(data)==200
    rows=[]
    for index,row in enumerate(data):
        output=generated[row['prompt_id']]
        rows.append(dict(index=index,prompt_id=row['prompt_id'],task=row['task'],
            input_tokens=row['input_tokens'],output_tokens=output['output_tokens'],
            cap_hit=output['cap_hit'],finish_reason=output['finish_reason'],
            output_ids_SHA256=__import__('hashlib').sha256(json.dumps(output['output_ids']).encode()).hexdigest()))
    result=dict(dataset_SHA256=digest(data_path),reference_job=job,reference_SHA256=digest(reference),
        n=200,input=stats([r['input_tokens'] for r in rows]),output=stats([r['output_tokens'] for r in rows]),
        input_output_ratio=stats([r['input_tokens']/r['output_tokens'] for r in rows]),
        cap_hits=sum(r['cap_hit'] for r in rows),by_task={})
    for task in sorted({r['task'] for r in rows}):
        part=[r for r in rows if r['task']==task]
        result['by_task'][task]=dict(n=len(part),input=stats([r['input_tokens'] for r in part]),
            output=stats([r['output_tokens'] for r in part]),cap_hits=sum(r['cap_hit'] for r in part))
    result.update(evidence='MEASURED_NATURAL_OUTPUT_REFERENCE',
        scope='Sequential standard vLLM outputs. Not open-loop serving performance. Stop tokens included in emitted output counts. Frozen inputs are unchanged.')
    write(ROOT/'analysis'/f'longbench200_natural_length_statistics{suffix}.json',result)
    write(ROOT/'data'/f'longbench200_measured_output_manifest{suffix}.json',dict(rows=rows,dataset_SHA256=digest(data_path),reference_SHA256=digest(reference)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tasks=list(result['by_task']);colors=plt.get_cmap('tab10').colors
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for task,color in zip(tasks,colors):
        part=[r for r in rows if r['task']==task]
        axes[0].hist([r['input_tokens'] for r in part],bins=np.linspace(0,40000,17),histtype='step',label=task,color=color)
        axes[1].hist([r['output_tokens'] for r in part],bins=np.linspace(0,2050,22),histtype='step',label=task,color=color)
    axes[0].set_xlabel('Actual Qwen input tokens');axes[1].set_xlabel('Natural generated output tokens')
    for ax in axes:ax.set_ylabel('Requests');ax.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.suptitle('Frozen LongBench v1: 5 tasks × 40 requests | Greedy, thinking disabled, cap2048')
    fig.tight_layout()
    for extension in ('png','pdf'):fig.savefig(ROOT/'figures'/f'longbench200_length_distribution{suffix}.{extension}',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.6))
    for task,color in zip(tasks,colors):
        part=[r for r in rows if r['task']==task]
        ax.scatter([r['input_tokens'] for r in part],[r['output_tokens'] for r in part],s=22,alpha=.65,label=task,color=color)
    ax.set(xlabel='Actual Qwen input tokens',ylabel='Natural generated output tokens',title='LongBench v1 frozen200 | Standard vLLM concurrency1 reference')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
    for extension in ('png','pdf'):fig.savefig(ROOT/'figures'/f'longbench200_input_output_joint{suffix}.{extension}',dpi=160)
    plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
