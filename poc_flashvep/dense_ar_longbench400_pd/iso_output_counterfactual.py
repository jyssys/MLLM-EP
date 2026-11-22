"""Change only the controlled output contract at the exact iso8 Output64 load."""
import argparse
import json
import time
from .common import ROOT,write,digest
from .run import save
from .confirmation_analysis import load,ratio
from .service_accounting import analyze


MANIFEST='iso8_I3_exact_output_counterfactual_v1.json'


def plan():
    original=json.loads((ROOT/'configs/iso8_hybrid_confirmation_frozen_v1.json').read_text())
    cells=[r for r in original['cells'] if r['input_bucket']==3 and r['output']==64]
    jobs=[]
    rows=[]
    assert len(cells)==3
    for index,cell in enumerate(cells):
        pair={}
        for axis in ('Native','Layer','Token','Hybrid'):
            cfg=json.loads((ROOT/'configs'/(cell[axis]+'.json')).read_text())
            assert cfg['output_cap']==64 and cfg['ignore_eos'] and cfg['controlled_pipeline']
            name=f'Iso8_I3_ExactOutput16_s{index}_{axis}_v1'
            cfg.update(name=name,output_cap=16,
                output_counterfactual_control=cell[axis],
                scope='Same input IDs,exact source capsule,GPU pair,arrival offsets,cut,credit and B/S as the Output64 control. '
                    'Change only controlled output64 to16. Natural EOS serving is not represented. '
                    'This removes the0.16percent QPS difference in the earlier two output contracts. '
                    'Keep every outcome. The aim is causal discrimination,not a repeat-until-win extension.')
            save(cfg)
            jobs.append(name)
            pair[axis]=dict(output16=name,output64=cell[axis])
        rows.append(dict(start=index,QPS=cell['QPS'],jobs=pair))
    path=ROOT/'configs'/MANIFEST
    assert not path.exists()
    write(path,dict(jobs=jobs,cells=rows,source_capsule_SHA256=original['source_capsule_SHA256'],
        evidence='FROZEN_OUTPUT_ONLY_CAUSAL_COUNTERFACTUAL',
        competing_predictions=['Longer decode residency can make the destination queue dominate even when source service is slower.',
            'If output residency is not decisive,the matched-output change need not reverse the axis ordering.'],
        scope='Three independent paired start blocks at one exact QPS. The existing Output64 starts are retained,not repeated. '
            'These are C0 actual1/1/1 controls. They do not establish continuous-serving superiority.'))
    print('EXACT_OUTPUT_COUNTERFACTUAL_PLANNED',len(jobs),flush=True)


def analyze_all():
    contract=json.loads((ROOT/'configs'/MANIFEST).read_text())
    values={o:{a:[] for a in ('Native','Layer','Token','Hybrid')} for o in (16,64)}
    stages={o:{a:[] for a in ('Native','Layer','Token','Hybrid')} for o in (16,64)}
    for cell in contract['cells']:
        block={}
        for axis,work in cell['jobs'].items():
            block[axis]={}
            for output in (16,64):
                name=work['output'+str(output)]
                m,cfg=load(name,contract['source_capsule_SHA256'])
                if m is None:
                    return dict(status='NOT_RUN')
                assert m['output_tokens']==16*output and cfg['output_cap']==output
                block[axis][output]=cfg
                values[output][axis].append(m)
                stages[output][axis].append(analyze(name)['means'])
            a,b=block[axis][16],block[axis][64]
            for field in ('dataset_SHA256','arrival_offsets','seed','gpus','axis_cut',
                    'hidden_source_slots','hidden_destination_slots','token_budget','max_num_seqs'):
                assert a.get(field)==b.get(field),(axis,field)
    rows=[]
    for output in (16,64):
        tests={}
        for ref in ('Layer','Token','Native'):
            tests[ref]={f:ratio([r[f] for r in values[output]['Hybrid']],
                [r[f] for r in values[output][ref]]) for f in
                ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_qualified_output_tokens_per_second')}
        deltas={ref:{field:sum(a[field]-b[field] for a,b in zip(stages[output]['Hybrid'],stages[output][ref]))/3
            for field in stages[output]['Hybrid'][0]} for ref in ('Layer','Token')}
        rows.append(dict(controlled_output=output,Hybrid_over=tests,
            additive_mean_host_Hybrid_minus_control_ms=deltas,
            Hybrid_native_decode_5percent=all(tests['Native'][f]['CI95'][1]<=1.05
                for f in ('ITL_p99_ms','TPOT_p95_ms'))))
    result=dict(time=time.time(),status='MEASURED_ONLINE',cells=rows,
        source_capsule_SHA256=contract['source_capsule_SHA256'],
        evidence='MEASURED_ONLINE_WITH_TRACE_ANALYSIS',
        scope='Exact output-only intervention at one QPS with three independent server-start blocks. '
            'Pointwise paired empirical95percent CIs,27 resamples. '
            'Mean host-boundary differences do not decompose TTFTp95 or exclusive GPU time. '
            'C0 whole-request destination admission and source credit4 remain part of this result. '
            'Do not transfer this causal result to unrestricted continuous batching or natural EOS without testing.')
    write(ROOT/'analysis/iso8_I3_exact_output_counterfactual_v1.json',result)
    print([(r['controlled_output'],r['Hybrid_over']['Token']['TTFT_p95_ms']) for r in rows],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--prepare-queue',action='store_true')
    args=parser.parse_args()
    if args.plan:
        plan()
    elif args.prepare_queue:
        source=ROOT/'configs'/MANIFEST
        cfg=json.loads(source.read_text())
        destination=ROOT/'jobs'/MANIFEST
        assert not destination.exists()
        write(destination,dict(jobs=cfg['jobs'],frozen_config_manifest_SHA256=digest(source),
            scope='Queue receipt for the existing immutable output-only experiment design.'))
    else:
        analyze_all()


if __name__=='__main__':
    main()
