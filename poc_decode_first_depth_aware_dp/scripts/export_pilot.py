"""Publish only actual Stage-0 trace events and descriptive quality figures."""
import argparse
import collections
import csv
import gzip
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_decode_first_depth_aware_dp.contract import *


def table(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output
    out.mkdir(exist_ok=False,parents=True);figures=out/'figures';figures.mkdir()
    summary=read(args.analysis/'quality_summary.json')
    scored=read(args.analysis/'quality_predictions.json');hist=collections.Counter();requests=[];count=0
    with gzip.open(out/'decode_depth_traces.jsonl.gz','wt') as stream:
        for row in scored:
            raw=read(row['raw_path']);depths=[]
            for event in raw['depth_events']:
                value=dict(event,request_id=raw['id'],dataset=raw['dataset'],policy=raw['policy'],
                    prompt_tokens=raw['prompt_tokens'],actual_output_length=raw['output_length'],
                    gate_q=.5 if raw['policy']=='dynamic' else None,KV_policy='last_exited',
                    evidence='MEASURED_ONLINE_STAGE0_QUALITY_PILOT. NOT_A_STAGE1_SERVING_COHORT.')
                assert event['output_token_index']>0
                stream.write(json.dumps(value)+'\n');depths.append(event['exit_depth_executed']);count+=1
                hist[raw['policy'],raw['dataset'],event['exit_depth_executed']]+=1
            requests.append(dict(request_id=raw['id'],dataset=raw['dataset'],policy=raw['policy'],
                prompt_tokens=raw['prompt_tokens'],output_tokens=raw['output_length'],
                decode_tokens=len(depths),mean_depth=statistics.mean(depths) if depths else None,
                within_request_depth_variance=statistics.pvariance(depths) if depths else None,
                total_executed_loop_work=sum(depths),wall_ms=raw['wall_ms'],
                TTFT_ms=raw['TTFT_ms'],decode_span_host_ms=raw['decode_span_ms'],
                finish_reason=raw['finish_reason'],physical_gpu=raw['depth_events'][0]['physical_gpu'] if depths else None,
                scope='Measured quality observer wall. Not pure GPU service time or live DP scheduling performance.'))
    table(out/'request_depth_summary.csv',requests)
    table(out/'depth_histogram.csv',[dict(policy=p,dataset=d,executed_depth=k,tokens=v) for (p,d,k),v in sorted(hist.items())])
    plt.figure(figsize=(7,4))
    for i,row in enumerate(summary['rows']):
        low,high=row['paired_bootstrap_CI95_drop_pp'];point=row['accuracy_drop_pp']
        plt.errorbar([i],[point],yerr=[[point-low],[high-point]],fmt='o',capsize=6)
    plt.xticks([0,1],['GSM8K','MATH-500']);plt.axhline(0,color='k',ls='--')
    plt.ylabel('Fixed minus dynamic accuracy, percentage points')
    counts=', '.join(f"{r['dataset']} n={r['n']}" for r in summary['rows'])
    plt.title('Stage-0 paired request-bootstrap 95% CI\n'+counts)
    plt.tight_layout();plt.savefig(figures/'paired_accuracy_drop.png',dpi=160);plt.close()
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,dataset in zip(axes,['gsm8k','math500']):
        total=sum(hist['dynamic',dataset,k] for k in range(1,5))
        ax.bar(range(1,5),[100*hist['dynamic',dataset,k]/total if total else 0 for k in range(1,5)])
        ax.set(xticks=range(1,5),xlabel='Actual executed decode depth',ylabel='Observed token fraction, %',title=dataset)
    fig.suptitle('Online gate q=0.5, minimum depth 2. Prefill-produced token excluded.')
    fig.tight_layout();fig.savefig(figures/'online_depth_histogram.png',dpi=160);plt.close(fig)
    plt.figure(figsize=(7,4))
    for d in ['gsm8k','math500']:
        rows=[r for r in requests if r['dataset']==d and r['policy']=='dynamic' and r['mean_depth'] is not None]
        plt.scatter([r['output_tokens'] for r in rows],[r['mean_depth'] for r in rows],label=d)
    plt.xlabel('Actual natural output tokens');plt.ylabel('Request mean executed decode depth');plt.legend()
    plt.title('Pilot description only. Not length-controlled or admission prediction.')
    plt.tight_layout();plt.savefig(figures/'request_depth_vs_length.png',dpi=160);plt.close()
    write(out/'trace_manifest.json',dict(actual_decode_events=count,requests=len(requests),
        distinct_questions=len(requests)//2,domains=['gsm8k','math500'],first_output_token_excluded=True,
        source_quality_summary_sha256=sha(args.analysis/'quality_summary.json'),
        model=MODEL_ID,revision=REVISION,CDB_revision=CDB_REVISION,
        no_replayed_depths=True,no_tokens_forced_after_EOS=True,stage='STAGE_0_QUALITY_PILOT_ONLY',
        serving_cost_model_validated=False,DP4_placement_baselines_run=False,
        future_depth_oracle_run=False,RL=False))
    print('EXPORTED_ACTUAL_STAGE0_EVENTS',count)


if __name__=='__main__':main()
