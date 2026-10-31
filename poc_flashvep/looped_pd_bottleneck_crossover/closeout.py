"""Write final review tables from completed evidence. Start no GPU work."""
import collections
import csv
import datetime
import gzip
import hashlib
import shutil
from .common import *
from .summary_tables import save_csv


def read(name):return list(csv.DictReader((ROOT/name).open()))


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in rows])+'\n'


def number(value,digits=3):return f'{float(value):.{digits}f}'


def tables():
    runs=read('serving_runs.csv');clean=[r for r in runs if r['eligible_for_serving_comparison']=='True']
    screen=[r for r in clean if '_screen_start0' in r['name']]
    text='# Completed Native PD Screens\n\n'
    text+='Evidence: MEASURED. All latencies use milliseconds. Goodput uses requests/s.\n'
    text+='Each row uses one independent start. The Looped rows use C256 and default pools.\n'
    text+='The Qwen rows use native vLLM. They are not matched-kernel architecture comparisons.\n\n'
    text+=table(['Model','Cell','N','Target rate','Realized rate','Joint passed/N','TTFT p95','TPOT p95','ITL p99','E2E p95','Finite goodput'],[
        [r['model'],r['cell'],r['offered_requests'],number(r['arrival_rate']),number(r['realized_arrival_rate']),
        r['joint_SLO_passed']+'/'+r['offered_requests'],*[number(r[k],2) for k in ('TTFT_ms_p95','request_TPOT_ms_p95','single_token_delivery_ITL_ms_p99','E2E_ms_p95')],number(r['SLO_goodput_requests_per_s'],4)] for r in screen])
    text+='\nThese finite observations do not establish sustainable capacity.\n'
    (ROOT/'completed_screen_tables.md').write_text(text)
    rates=[r for r in clean if '_rate' in r['name']]
    text='# Completed G2 Rate Sweeps\n\nEvidence: MEASURED. G2 is 3,072 input tokens and 128 controlled output tokens.\n'
    text+='All rows use C256 and the default pool. Each row has 64 requests.\n'
    text+='The factor 0.8 has three independent starts per model.\n\n'
    text+=table(['Model','Factor','Start','Target requests/s','Joint passed/N','TTFT p95 ms','TPOT p95 ms','E2E p95 ms','Finite goodput requests/s'],[
        [r['model'],r['rate_factor'],r['independent_start'],number(r['arrival_rate'],4),r['joint_SLO_passed']+'/64',
        *[number(r[k],2) for k in ('TTFT_ms_p95','request_TPOT_ms_p95','E2E_ms_p95')],number(r['SLO_goodput_requests_per_s'],4)] for r in rates])
    text+='\nNo measured point in these C256 sweeps reaches 90% joint attainment.\n'
    text+='This does not locate a zero capacity. Lower rates and tuned-engine sweeps remain unmeasured.\n'
    (ROOT/'completed_rate_sweep_tables.md').write_text(text)
    repeats=[r for r in read('independent_start_summary.csv') if int(r['independent_starts'])>=3]
    def interval(r,k):return number(r[k+'_mean'],4)+' ['+number(r[k+'_between_start_CI95_low'],4)+', '+number(r[k+'_between_start_CI95_high'],4)+']'
    text='# Independent-start Uncertainty\n\nEvidence: MEASURED runs with TRACE_ANALYSIS intervals. Each setting has three independent starts and 64 requests per start.\n'
    text+='The intervals use Student-t with two degrees of freedom. Bounds intersect the valid parameter range.\n'
    text+='A p95 entry summarizes the mean of three per-start p95 values. It is not a pooled p95.\n\n'
    text+=table(['Model','Cell','Chunk / pool / slots','Target requests/s','Goodput mean [95% CI]','Attainment mean [95% CI]','TPOT p95 mean [95% CI], ms'],[
        [r['model'],r['cell'],r['chunk']+' / '+r['KV_blocks']+' / '+r['compute_slots'],number(r['target_rate'],4),
        interval(r,'SLO_goodput_requests_per_s'),interval(r,'joint_SLO_attainment'),interval(r,'request_TPOT_ms_p95')] for r in repeats])
    text+='\nThese are setting-level intervals, not paired method-improvement intervals.\n'
    (ROOT/'independent_start_intervals.md').write_text(text)
    names=['phase_b_ouro_g2_screen_start0','phase_b_ouro_g2_tuned_chunk_start0',
        'phase_b_huginn_g2_screen_start0','phase_b_huginn_g2_tuned_chunk_retry_start0']
    shares=read('e2e_component_shares_aggregate.csv')
    components=list(dict.fromkeys(r['component'] for r in shares if r['name'] in names))
    text='# Host E2E Component Percentages\n\nEvidence: MEASURED_HOST_EXCLUSIVE_INTERVALS. Each setting has 24 requests at G2.\n'
    text+='Each column divides summed component time by summed request E2E. It totals 100% before display rounding.\n'
    text+='The columns are not GPU-kernel or DMA decompositions.\n\n'
    text+=table(['Component','Ouro C256 %','Ouro C2048 %','Huginn C256 %','Huginn C2048 %'],[
        [component]+[number(next(r['share_percent'] for r in shares if r['name']==name and r['component']==component),3) for name in names] for component in components])
    text+='\nA Prefill service interval can contain queueing and host control work. It is not pure computation.\n'
    text+='Cumulative transfer completion time can overlap computation. It cannot form another additive percentage.\n'
    (ROOT/'host_e2e_component_table.md').write_text(text)


def review():
    runs=read('serving_runs.csv');clean=[r for r in runs if r['eligible_for_serving_comparison']=='True']
    assert len(clean)==56 and sum(int(r['completed_requests']) for r in clean)==2544
    assert all(r['failed_or_missing']=='0' for r in clean)
    screening=read('bottleneck_screening.csv')
    candidates=[r['name'] for r in screening if r['P_candidate']=='True']
    assert candidates==['phase_b_ouro_g1_screen_start0'],candidates
    for model in ('ouro','huginn'):
        assert sum(r['model']==model and '_screen_start0' in r['name'] for r in clean)==8
        assert sum(r['model']==model and '_rate' in r['name'] for r in clean)==7
    parity=json.loads((ROOT/'state/cross_run_output_parity.json').read_text())
    assert not parity['failures'] and parity['exact_groups']==parity['matched_input_seed_groups']==264
    assert all(r['exact']=='True' for r in read('fixed_depth_work_accounting.csv'))
    decisions=dict(time=time.time(),review='FINAL_EVIDENCE_REVIEW',provisional=False,
        primary_grid='G1-G6 and C1/C2. D32 remains a secondary diagnostic.',
        architectural_causality='INCONCLUSIVE_NOT_A_LOOPED_MODEL_IMPOSSIBILITY_RESULT',
        untested_controls=['Whole-prompt atomic Full-PD','Exact 4K/100','Ouro 8K/100 or 8K/200',
            'Faithful Qwen3-8B 8000/200 DuetServe setting'],
        models={m:dict(characterization='COMPLETED_FINITE_SCOPE',
            prefill_crossover='NO_P_HEADROOM_ON_TESTED_GRID',
            phase_C='NOT_RUN_GATE_NOT_MET',new_online_controller='NOT_IMPLEMENTED_GATE_NOT_MET',
            reason='The untuned finite P candidate is not validated beyond generic tuning and profiling limits.' if m=='ouro' else
                'Observed D KV admission pressure prevents validation of D spare capacity.',
            original_HF_accuracy='NOT_MEASURED',universal_dynamic_A_limit='NOT_ESTABLISHED') for m in ('ouro','huginn')},
        numerical_qualification='PASS_INTERNAL_FINITE_CHECKS_NOT_ORIGINAL_MODEL_EQUIVALENCE',
        uncertainty='Clean compute/host causality and infinite-horizon stability remain unresolved.',
        source_gate='The earlier provisional gate remains preserved in primary_phase_c_gate.json.')
    write(ROOT/'state/final_phase_decisions.json',decisions)
    rows=[]
    for model in ('ouro','huginn'):
        for method in ('DP2','DP2_Chunked','PD_G1','Full_cut_snapshot','Loop_Static','Loop_Dynamic','Token_Static','Token_Dynamic'):
            measured=method in ('DP2','DP2_Chunked','PD_G1')
            rows.append(dict(model=model,method=method,phase_C_status='NOT_RUN_GATE_NOT_MET',
                phase_B_control='MEASURED_G2' if measured else 'NOT_RUN_IN_THIS_CAMPAIGN',
                new_partition_work='NOT_RUN',online_method_speedup='NOT_MEASURED',
                scope='A measured native full-cut baseline is not a failed new Dynamic policy.'))
    save_csv(ROOT/'phase_c_method_status.csv',rows)


def compress_derived():
    source=ROOT/'per_request_analysis.json';target=ROOT/'per_request_analysis.json.gz'
    with source.open('rb') as inp,gzip.open(target,'wb',compresslevel=6) as out:shutil.copyfileobj(inp,out)
    expected=digest(source);h=hashlib.sha256()
    with gzip.open(target,'rb') as inp:
        while value:=inp.read(1024*1024):h.update(value)
    assert h.hexdigest()==expected
    write(ROOT/'state/derived_analysis_compression.json',dict(time=time.time(),source=source.name,
        published=target.name,uncompressed_SHA256=expected,compressed_SHA256=digest(target),
        uncompressed_bytes=source.stat().st_size,compressed_bytes=target.stat().st_size,
        scope='Lossless publication copy. The uncompressed local file remains intact.'))


def main():
    review();tables();compress_derived();print('CLOSEOUT_TABLES_AND_REVIEW_COMPLETE')


if __name__=='__main__':main()
