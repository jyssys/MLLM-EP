"""Write evidence-linked status reports. An incomplete campaign receives no verdict."""
import csv
import datetime
from .contract import *


def csv_rows(name):
    path=ROOT/name
    return list(csv.DictReader(path.open())) if path.exists() else []


def fmt(value,digits=3):
    if value in [None,'','None']:return 'NOT_RUN'
    return f'{float(value):.{digits}f}'


def md_table(header,rows):
    return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']+
        ['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'


def save(name,text):
    (ROOT/name).write_text(text.rstrip()+'\n')


def main():
    accuracy=csv_rows('accuracy_by_arm.csv');serving=csv_rows('subset400_serving.csv');replay=csv_rows('replay_summary.csv')
    pairs=csv_rows('paired_accuracy_deltas.csv');byquality={(r['run'],r['dataset']):r for r in accuracy}
    status=read(ROOT/'state/dynamic_probe_status.json') if (ROOT/'state/dynamic_probe_status.json').exists() else {}
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();quality=[];matrix=[]
    for arm,config in ARMS.items():
        counts=[]
        for domain,target in [('gsm8k',300),('math500',100)]:
            r=byquality.get(('quality400_'+arm,domain))
            counts.append(int(r['N']) if r else 0)
            def delta(reference):
                match=next((v for v in pairs if v['run']=='quality400_'+arm and v['reference']==reference and v['dataset']==domain),None)
                if not match:return 'NOT_RUN'
                return f"{fmt(match['delta_accuracy_pp'])} [{fmt(match['CI95_low_pp'])}, {fmt(match['CI95_high_pp'])}]"
            if r:quality.append([domain,arm,f"{r['correct']}/{r['N']}",fmt(r['accuracy_percent']),
                delta('quality400_S4'),delta('quality400_SD' if config['mode']=='dynamic' else 'quality400_S4'),
                fmt(r['avg_output_tokens']),fmt(r.get('mean_decode_depth')),r['truncated'],r['status']])
            else:quality.append([domain,arm,'0/'+str(target),'NOT_RUN','NOT_RUN','NOT_RUN','NOT_RUN','NOT_RUN','NOT_RUN',status.get(arm,{}).get('status','NOT_RUN')])
        completion='TASK_TARGET_COMPLETE' if counts==[300,100] else status.get(arm,{}).get('status','IN_PROGRESS')
        matrix.append([arm,config['parallelism'],config['mode'],','.join(map(str,config['GPUs'])),f'{counts[0]}/300 + {counts[1]}/100',
            completion,'NUMERICALLY_UNQUALIFIED_EXPLORATORY' if arm in ['T4','TD'] else 'See correctness.md'])
    quality_text=md_table(['Dataset','Arm','Correct/N','Accuracy %','Delta vs S4 pp [95% CI]',
        'Delta vs mode Single pp [95% CI]','Mean output tokens','Token-weighted decode depth','Truncated','Fidelity'],quality)
    native=[]
    for name in ['native_TP1','native_TP4']:
        for domain in ['gsm8k','math500']:
            r=byquality.get(('quality400_'+name,domain))
            if r:native.append([domain,name,f"{r['correct']}/{r['N']}",fmt(r['accuracy_percent']),r['truncated'],r['status']])
    native_text=md_table(['Dataset','Native engine','Correct/N','Accuracy %','Truncated','Fidelity'],native)
    save('eight_arm_matrix.md','# Eight-arm status\n\nUpdated UTC: '+stamp+'.\n\n'+md_table(
        ['Arm','Parallelism','Mode','Physical GPUs','Evaluated target','Execution status','Fidelity'],matrix)+
        '\nThe target is 400 questions per arm. The complete 1819-question suites are USER_REDUCED_NOT_RUN.\n'+
        'A completed task target does not pass the historical numerical gate.\n\n'+quality_text)
    task_table=md_table(['Run','N','Cohort sum s','Mean request E2E s','Mean TTFT ms','Mean TPOT ms','Output tok/s','Status'],[
        [r['run'],r['N'],fmt(r['cohort_makespan_sum_s']),fmt(r['mean_E2E_s']),
         fmt(float(r['mean_TTFT_s'])*1000 if r['mean_TTFT_s'] else None),
         fmt(float(r['mean_TPOT_s'])*1000 if r['mean_TPOT_s'] else None),fmt(r['output_tokens_per_s']),r['status']] for r in serving])
    replay_text=md_table(['Family','Case','Arm','C','Repeats','Median makespan s','Speedup vs mode Single','Overall vs S4','Status'],[
        [r['family'],r['case'],r['arm'],r['C'],r['measured_repeats'],fmt(r['median_cohort_makespan_s']),
         fmt(r['speedup_cohort_vs_mode_Single'])+'x' if r['speedup_cohort_vs_mode_Single'] else 'NOT_RUN',
         fmt(r['overall_cohort_speedup_vs_S4'])+'x' if r['overall_cohort_speedup_vs_S4'] else 'NOT_RUN',r['status']] for r in replay]) if replay else 'NOT_RUN. No completed clean replay comparison is available.\n'
    save('benchmark_summary.md',f'''# Benchmark summary

Updated UTC: {stamp}.
This is an evidence snapshot. Missing experiments remain NOT_RUN.

## Actual answer accuracy

{quality_text}
The same 300 GSM8K and 100 MATH question IDs define each final target.
The current denominator is the completed, canonical-input subset, not the intended target.
The sample preserves the original pilot and adds metadata-only random questions.
All accuracy uses actual greedy generation, EOS, and a 4096-token cap.
Token-weighted decode depth excludes the first token, which comes from full R4 prefill.
Answer accuracy includes correctly parsed truncated answers. The CSV reports completed-correct answers separately.
Paired intervals use question-level bootstrap with 5000 draws and seed 20261021.
Small intervals from identical observed answers do not prove population equivalence.

## Native production-oriented Static controls

{native_text}
The frozen arm manifest prefers native vLLM TP4 as the production-oriented T4 reference.
The custom eight-arm table provides the compatibility-controlled diagnostic family.
These native controls use the same 400-question target. They are not additional depth policies.
Do not mix native batching/kernel gains with a loop-placement gain.

## Task-observer timing

{task_table}
These times include task observers and controller work. They are not observer-free hardware headlines.
The custom service clock excludes request-bound graph preparation. It is not cold-cohort E2E.
Later raw records separately measure resident-model preparation-inclusive cohort wall time.
Earlier custom TP pilot records lack that field. Preparation-inclusive aggregate coverage remains PARTIAL until all source cohorts expose it.
The cohort sum excludes setup between cohorts. It is not end-to-end wall time for the entire experiment command.
Static custom PP uses asynchronous request FIFO. Custom TP uses token waves.
Those Static task timings are not fully scheduler-matched.
Native vLLM has different batching, attention, graph, and KV implementations.
Keep the custom eight-arm comparison separate from the two native production-oriented controls.
Generated token counts can differ between backends. They remain visible in `subset400_serving.csv`.

## Identical-work clean replay

{replay_text}
Replay uses frozen output IDs, not benchmark answers. Dynamic replay also locks observed Single-Dynamic exit depths.
Every replay excludes setup and warm-up repeat zero. The plan requests five measured repeats.
Numerically unresolved TP timing remains exploratory.
`replay_summary.csv` separates request-paired E2E intervals from weak five-repeat cohort intervals.
''')
    probes=[]
    for arm in ['L4','O4','LD','OD']:
        label='probe_'+arm+('_v3' if arm.endswith('4') else '_live_v1')
        file=ROOT/'numerical'/f'probe_{label}.json'
        if file.exists():
            rows=read(file);probes.append([arm,len(rows),sum(r['token_exact'] for r in rows),sum(r['depths_exact'] for r in rows),
                sum(r['KV_bitwise'] for r in rows),'All 96 logical slots per request'])
        else:probes.append([arm,'NOT_RUN','NOT_RUN','NOT_RUN','NOT_RUN','Pending physical probe'])
    save('correctness.md',f'''# Correctness and validity

Updated UTC: {stamp}.

## Independent axes

The historical P512 strict numerical gate remains FAIL.
No new benchmark-derived numerical tolerance replaces it.
TP4 task accuracy is NUMERICALLY_UNQUALIFIED_EXPLORATORY.
Valid vocabulary IDs and finite logits permit this exploratory task lane. They do not establish tensor equivalence.

{md_table(['Arm','Probe requests','Exact output','Exact depths','Bitwise KV','KV scope'],probes)}
The PP comparisons use the same custom Single backend and actual owner-local KV tensors.
Forced-depth probes and live-gate probes remain different lanes.
The official CDB versus custom comparison, when present, is `numerical/dynamic_reference_comparison.json`.
Its backend warning must remain visible if outputs or gate decisions differ.
The official cache-copy checks and prototype tests independently check last_exited current-row semantics.

## Preserved failures

The initial new Graph path had an input-lifetime error. The corrected path retains the mask input tensor.
This correction does not explain the historical eager TP4 numerical failure.
An older Single task record used different whitespace tokenization. It remains preserved and excluded from canonical paired comparisons.
The first native TP1 task run has invalid TTFT/TPOT. It is not the timing baseline.
Earlier raw logs and failed probes remain available. See `failures.md`.

## Scope limits

Probe parity does not prove long-generation quality equivalence.
The quality CSV measures actual answers separately.
CUDA kernel service sums are not exclusive E2E component shares.
Native setup memory is not a measured full-task HBM peak.
''')
    stage=read(ROOT/'state/continuation_stage.json') if (ROOT/'state/continuation_stage.json').exists() else {'stage':'Static pilot'}
    save('analysis.md',f'''# Eight-arm discovery analysis

Updated UTC: {stamp}.

## Current decision

The campaign is IN_PROGRESS. No final architecture GO/NO-GO verdict is issued in this snapshot.
The current controller stage is `{stage.get('stage')}`.
`eight_arm_matrix.md` lists completed samples and remaining execution states.

## What the numerical evidence supports

Real TP4 exists. The model has sharded attention/FFN weights and actual four-rank collectives.
The old bounded operator audit passed 192/192 operations. The historical full-model P512 gate remains FAIL.
The new five-engine synthetic study contains 112 teacher-aligned positions across eight synthetic requests.
Every BF16 pair agrees on argmax at those positions. Native TP1 already differs from HF internally.
These observations do not prove benchmark accuracy or identify one unique numerical root cause.
See `numerical_tp_diagnosis.md`, `first_divergence.csv`, and `logit_metrics.csv`.

## What the serving evidence must distinguish

Loop PP reduces physical stage handoffs relative to re-entrant Layer PP.
That count reduction is structural accounting, not a measured E2E speedup.
The primary controlled replay uses common no-refill waves and identical token work.
The additional matched-natural cohort uses the same successor rule on all eight arms.
The larger Static task cohort preserves legacy asynchronous PP and therefore has a scheduling confound against TP.
The native vLLM controls remain separate from custom backend comparisons.
A four-GPU ratio against Single uses four times the GPU count. It is not a claim of four-GPU resource efficiency.

## Dynamic interpretation

The locked method uses the original CDB cumulative-hazard gate, q=0.5, R2-R4, full-depth prefill, and last_exited KV.
The prototype must physically skip later loops after exit.
Current exit KV rows must reach every deeper owner before the next token starts.
Loop PP can therefore add exit-KV communication that does not exist in layer-local filling.
This is a prospective physical hypothesis. Its latency contribution requires actual timing.
Do not mix this method with another paper's gate or KV policy.

## Completion requirements

The final review must check all eight quality rows, numerical warnings, dynamic reference checks, five-repeat replay, and actual overlap timelines.
It must report blocked arms rather than invent results.
Only measured matched serving evidence can support an architecture conclusion.
Full public-suite evaluation is USER_REDUCED_NOT_RUN under the user's 400-question amendment.
''')
    save('status.md',f'# Study status\n\nUpdated UTC: {stamp}.\n\n'+md_table(['Arm','Parallelism','Mode','GPUs','Completed target','Status','Fidelity'],matrix)+
        '\nThe controller can continue independently of the chat turn. Inspect `state/continuation_controller.json` and `state/continuation_stage.json`.\n')
    print('REPORT_SNAPSHOT',stamp,flush=True)


if __name__=='__main__':main()
