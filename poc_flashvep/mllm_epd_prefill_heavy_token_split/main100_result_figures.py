"""Render frozen Main100 distributions and measured request-level failures."""
from collections import Counter
import csv
import json
import time

import numpy as np

from .common import ROOT,digest,preserve,write
from .token_confirmation_analysis import CATEGORIES
from .token_discovery_analysis import table


def dataset_rows(manifest,reference):
    lookup={r['request_id']:r for r in reference}
    assert len(lookup)==len(reference)
    result=[]
    for row in manifest['requests']:
        actual=lookup[row['request_id']]
        assert actual['status']=='MEASURED_ONLINE' and actual['prompt_SHA256']==row['prompt_SHA256']
        assert row['total_input_tokens']==row['visual_tokens']+row['text_tokens']+row['special_tokens']
        assert actual['usage']['prompt_tokens']==row['total_input_tokens']
        assert actual['output_tokens']==len(actual['token_ids'])==actual['usage']['completion_tokens']
        result.append(dict(request_id=row['request_id'],source=row['source'],input_tokens=row['total_input_tokens'],
            visual_tokens=row['visual_tokens'],text_tokens=row['text_tokens'],special_tokens=row['special_tokens'],
            image_count=row['image_count'],page_count=row['page_count'],
            natural_output_tokens=actual['output_tokens'],cap_hit=actual['output_cap_hit'],
            visual_token_ratio=row['visual_tokens']/row['total_input_tokens'],
            input_output_ratio=row['total_input_tokens']/actual['output_tokens'],
            evidence='Frozen Main100 processor measurements plus original single-GPU Natural EOS length profile. Not C3 speedup.'))
    assert len(result)==len({r['request_id'] for r in result})==100
    return result


def failure_averages(rows):
    result=[]
    for mode in ['natural','replay']:
        for method in ['NativeStrong','NativeCommon','Static10','Idle10']:
            for category in CATEGORIES:
                selected=[r for r in rows if (r['mode'],r['method'],r['category'])==(mode,method,category)]
                assert len(selected)==3 and len({r['block'] for r in selected})==3
                result.append(dict(mode=mode,method=method,category=category,
                    start_mean_request_percent=float(np.mean([100*r['requests']/r['request_denominator'] for r in selected])),
                    start_mean_token_percent=float(np.mean([100*r['output_tokens']/r['total_output_tokens'] for r in selected])),
                    independent_starts=3,scope='Start-mean fractions. Stacked descriptions, not category-level CI or causal attribution.'))
    return result


def main():
    preserve();audit=ROOT/'analysis/main100_result_figures_v0.json';assert not audit.exists()
    manifest=ROOT/'data/main100_manifest_template_v2.json';data=json.loads(manifest.read_text())
    reference=ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl'
    reference_rows=[json.loads(line) for line in reference.read_text().splitlines()]
    rows=dataset_rows(data,reference_rows);table(ROOT/'analysis/main100_actual_length_rows_v0.csv',rows)
    statistics=[]
    for key in ['input_tokens','visual_tokens','text_tokens','special_tokens','natural_output_tokens','image_count']:
        values=np.array([r[key] for r in rows],dtype=float)
        statistics.append(dict(metric=key,requests=100,mean=float(values.mean()),median=float(np.median(values)),
            std_population=float(values.std()),minimum=float(values.min()),maximum=float(values.max()),
            **{f'p{p}':float(np.percentile(values,p)) for p in [10,25,75,90,95,99]}))
    table(ROOT/'analysis/main100_actual_length_statistics_v0.csv',statistics)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    generated=[]
    def save(figure,name,records):
        figure.tight_layout()
        for extension in ['png','pdf']:figure.savefig(ROOT/'figures'/f'{name}.{extension}',dpi=180)
        plt.close(figure);table(ROOT/'figures'/f'{name}.csv',records);generated.append(name)
    figure,axes=plt.subplots(1,3,figsize=(12,3.8))
    for axis,key,label in zip(axes,['input_tokens','visual_tokens','text_tokens'],['Total LLM input','Visual input','Text input']):
        axis.hist([r[key] for r in rows],bins=16);axis.set_xlabel(label+' tokens');axis.set_ylabel('Requests');axis.grid(alpha=.2)
    figure.suptitle('Frozen Main100 | Actual processor tokens | Special tokens remain separate')
    save(figure,'main100_actual_input_distribution_v0',rows)
    figure,axis=plt.subplots(figsize=(7,4))
    axis.hist([r['natural_output_tokens'] for r in rows],bins=20)
    axis.set_xlabel('Natural output tokens, including EOS');axis.set_ylabel('Requests');axis.grid(alpha=.2)
    axis.set_title('Main100 | Single-GPU Natural EOS profile | Cap2048, no forced length')
    save(figure,'main100_actual_natural_output_v0',rows)
    figure,axis=plt.subplots(figsize=(7,4))
    for source in sorted({r['source'] for r in rows}):
        selected=[r for r in rows if r['source']==source]
        axis.scatter([r['input_tokens'] for r in selected],[r['natural_output_tokens'] for r in selected],s=22,label=source.split('/')[-1])
    axis.set_xlabel('Actual LLM input tokens');axis.set_ylabel('Natural output tokens, including EOS')
    axis.set_title('Frozen Main100 | Single-GPU output lengths, not production traffic')
    axis.legend(fontsize=8);axis.grid(alpha=.2);save(figure,'main100_actual_input_output_v0',rows)
    figure,axes=plt.subplots(1,2,figsize=(10,4))
    for axis,key,label in zip(axes,['image_count','page_count'],['Images per request','Document pages per request']):
        selected=[r for r in rows if r[key] is not None]
        axis.hist([r[key] for r in selected],bins=16);axis.set_xlabel(label);axis.set_ylabel('Requests')
        axis.set_title(f'Observed {len(selected)}/100 requests');axis.grid(alpha=.2)
    figure.suptitle('Main100 | Missing document page count is not a measured zero')
    save(figure,'main100_image_page_distribution_v0',rows)
    source=ROOT/'analysis/token_confirmation_C37_failure_categories_v0.csv'
    with source.open() as stream:
        records=[dict(block=int(r['block']),method=r['method'],mode=r['mode'],category=r['category'],
            requests=int(r['requests']),request_denominator=int(r['request_denominator']),
            output_tokens=int(r['output_tokens']),total_output_tokens=int(r['total_output_tokens']))
            for r in csv.DictReader(stream) if r['subset']=='Main100']
    averages=failure_averages(records);table(ROOT/'analysis/main100_C37_failure_startmeans_v0.csv',averages)
    figure,axes=plt.subplots(2,2,figsize=(13,8));methods=['NativeStrong','NativeCommon','Static10','Idle10']
    colors=plt.get_cmap('tab10').colors
    for column,mode in enumerate(['natural','replay']):
        for row_index,(key,label) in enumerate([('start_mean_request_percent','Requests (%)'),('start_mean_token_percent','Generated output tokens (%)')]):
            axis=axes[row_index,column];bottom=np.zeros(4)
            for index,category in enumerate(CATEGORIES):
                values=np.array([next(r[key] for r in averages if (r['mode'],r['method'],r['category'])==(mode,method,category)) for method in methods])
                axis.bar(range(4),values,bottom=bottom,color=colors[index],label=category);bottom+=values
            assert np.allclose(bottom,100.)
            axis.set_xticks(range(4),methods,rotation=15);axis.set_ylabel(label);axis.set_title(mode);axis.set_ylim(0,100)
    axes[0,0].legend(fontsize=7,ncol=2)
    figure.suptitle('MEASURED_ONLINE | Main100 QPS2 | N3 start-mean fractions | Quality pending')
    save(figure,'main100_C37_joint_failure_breakdown_v0',averages)
    host=ROOT/'analysis/token_confirmation_C37_request_host_paths_v0.csv';lookup={r['request_id']:r for r in rows}
    with host.open() as stream:
        stages=[r for r in csv.DictReader(stream) if r['subset']=='Main100' and r['method'] in ['NativeCommon','Static10']]
    points=[]
    for mode in ['natural','replay']:
        for method in ['NativeCommon','Static10']:
            for request_id,request in lookup.items():
                selected=[r for r in stages if (r['mode'],r['method'],r['request_id'])==(mode,method,request_id)]
                assert len(selected)==3 and len({r['block'] for r in selected})==3
                points.append(dict(mode=mode,method=method,request_id=request_id,source=request['source'],
                    input_tokens=request['input_tokens'],image_count=request['image_count'],visual_token_ratio=request['visual_token_ratio'],
                    P_HTTP_start_mean_ms=float(np.mean([float(r['P_HTTP_enclosure']) for r in selected])),
                    scope='Descriptive N3 mean per request. P HTTP includes queue, host, transfer and forward. Not exclusive GPU prefill or a visual-token causal effect.'))
    figure,axes=plt.subplots(1,2,figsize=(12,4))
    for axis,mode in zip(axes,['natural','replay']):
        for method in ['NativeCommon','Static10']:
            selected=[r for r in points if r['mode']==mode and r['method']==method]
            axis.scatter([r['visual_token_ratio'] for r in selected],[r['P_HTTP_start_mean_ms'] for r in selected],s=15,alpha=.65,label=method)
        axis.set_xlabel('Visual / total input token ratio');axis.set_ylabel('P HTTP enclosure mean (ms)');axis.set_title(mode);axis.grid(alpha=.2)
    axes[0].legend();figure.suptitle('DIAGNOSTIC host association | Input length, page count and queue remain confounders')
    save(figure,'main100_C37_visual_ratio_P_HTTP_v0',points)
    write(audit,dict(time=time.time(),manifest_SHA256=digest(manifest),reference_raw_SHA256=digest(reference),
        figures=generated,statistics=statistics,source_counts=dict(Counter(r['source'] for r in rows)),
        scope='Frozen Main100 descriptive statistics and actual C37 host/SLO results. No input modification or new serving measurement.',
        original_Main400='Unchanged',quality='PENDING',GPU_exclusive_visual_token_cost='NOT_MEASURED'))
    preserve();print('MAIN100_REQUEST_FIGURES_REGENERATED',len(generated),flush=True)


if __name__=='__main__':main()
