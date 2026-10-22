"""Offline execution-budget envelope, not a serving-speedup simulator."""
import csv
import gzip
import json
import math
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,write
from .surface_analysis import csv_out
from .legal_actions import Frontier,enumerate_actions

CHUNKS=(64,128,256,512,768,1024,1536,2048)
BUDGETS=(20,30,40,50,60,80,100,125,150,200,250)

def main():
    samples=defaultdict(list)
    with gzip.open(ROOT/'quantum_observations.csv.gz','rt') as stream:
        for row in csv.DictReader(stream):
            # Use training and calibration costs only. Do not select from test outcomes.
            if row['split'] not in ('TRAIN','CALIBRATION'):
                continue
            p,n,c=int(row['P']),int(row['N']),int(row['C'])
            rows,g,depth=int(row['rows']),int(row['G']),int(row['depth_start'])
            work=int(row['prefill_token_loop_work'])
            # Exclude mixed/decode rows and clipped final chunks from this envelope.
            if int(row['decode_rows']) or rows!=c or row['policy']=='mixed':
                continue
            if row['phase']=='prefill_prelude':
                category=('prelude',0,0)
            elif work==c*g and g>0:
                category=('core',g,depth)
            else:
                continue
            bucket=int(math.ceil(math.log2(max(1,int(row['context_max'])))))
            samples[(p,n,c,row['policy'],bucket,*category)].append(float(row['scheduling_impact_ms']))
    costs=[]
    for key,values in samples.items():
        p,n,c,policy,bucket,kind,g,depth=key
        costs.append(dict(P=p,N=n,C=c,policy=policy,context_log2_bucket=bucket,kind=kind,G=g,depth=depth,
            observations=len(values),median_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95))))
    csv_out(ROOT/'execution_budget_costs.csv',costs)
    table={k:float(np.median(v)) for k,v in samples.items() if len(v)>=3}
    candidates=[]
    for p,n in ((2033,4),(8177,8)):
        bucket=int(math.ceil(math.log2(p)))
        frontier=Frontier('diagnostic_prompt',p,0,0,2048,2048,True)
        legal={(a.C,a.G):a for a in enumerate_actions(frontier,CHUNKS)}
        for c in CHUNKS:
            if c>p:
                continue
            for policy,g in (('atomic',4),('staged4',4),('grouped',2),('grouped3',3),('split',1)):
                if (c,g) not in legal:
                    continue
                first=table.get((p,n,c,policy,bucket,'core',g,0))
                if first is None:
                    continue
                prelude=0. if policy=='atomic' else table.get((p,n,c,policy,bucket,'prelude',0,0))
                if prelude is None:
                    continue
                remainder=[]
                depth=g
                while depth<4:
                    actual=min(g,4-depth)
                    value=table.get((p,n,c,policy,bucket,'core',actual,depth))
                    if value is None:
                        break
                    remainder.append(value);depth+=actual
                if depth!=4:
                    continue
                total=prelude+first+sum(remainder)
                candidates.append(dict(P=p,N=n,C=c,G=g,policy=policy,context_log2_bucket=bucket,
                    initial_quantum_plus_prelude_ms=prelude+first,estimated_full_chunk_ms=total,
                    immediate_token_loop_work=c*g,immediate_completed_prefill_tokens=c if g==4 else 0,
                    amortized_completed_tokens_per_ms=c/total,
                    evidence='OFFLINE_COARSE_CONTEXT_MEDIAN_COST_ENVELOPE'))
    csv_out(ROOT/'execution_budget_candidates.csv',candidates)
    output=[]
    for p,n in ((2033,4),(8177,8)):
        for budget in BUDGETS:
            feasible=[r for r in candidates if r['P']==p and r['N']==n and r['initial_quantum_plus_prelude_ms']<=budget]
            for family,allowed in [('Atomic_C_only',('atomic',)),('Best_G4_paths',('atomic','staged4')),('G_only_C1024',('split','grouped','grouped3','staged4')),('Joint',('atomic','staged4','split','grouped','grouped3'))]:
                menu=[r for r in feasible if r['policy'] in allowed and (family!='G_only_C1024' or r['C']==1024)]
                best=max(menu,key=lambda r:r['amortized_completed_tokens_per_ms']) if menu else None
                output.append(dict(P=p,N=n,budget_ms=budget,family=family,chosen_policy=best['policy'] if best else 'NONE',
                    chosen_C=best['C'] if best else 0,chosen_G=best['G'] if best else 0,
                    amortized_tokens_per_ms=best['amortized_completed_tokens_per_ms'] if best else 0.,
                    immediate_completed_prefill_tokens=best['immediate_completed_prefill_tokens'] if best else 0,
                    evidence='MODELED_EXECUTION_BUDGET_NOT_DEADLINE_OR_E2E'))
    csv_out(ROOT/'joint_headroom.csv',output)
    actual=json.loads((ROOT/'predictor_blind_results.json').read_text())
    selected=actual['quantum']
    strata=[r for r in selected['strata'] if r['stratum']=='phase' and r['key'] in ('atomic_prefill','staged_prefill')]
    fail=any(r['upper_bound_coverage']<.95 for r in strata)
    write(ROOT/'phase4_offline_decision.json',dict(phase=4,status='BLOCKED' if fail else 'CONDITIONAL',
        reason='Selected global margin misses prefill-specific coverage. The full-token predictor includes stock future interruptions and cannot serve as a candidate-specific decode DAG.',
        support='A legal single-chunk cost envelope exists. It does not establish queue-level joint opportunity or capacity.',
        online_phase5='NOT_RUN',qualified_real_state_opportunity_frequency='NOT_AVAILABLE',
        remaining_dependencies=['published GPU-event readiness at the pre-selection boundary','candidate-specific decode-only completion model','independently validated bounded schedule replay'],
        forbidden_claims=['modeled envelope is measured speedup','partial token-loop work is completed prompt progress','stock trace future completion is an online feature']))
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,p,n in zip(axes,(2033,8177),(4,8)):
        for family in ('Atomic_C_only','Best_G4_paths','G_only_C1024','Joint'):
            values=[r for r in output if r['P']==p and r['family']==family]
            ax.plot([r['budget_ms'] for r in values],[r['amortized_tokens_per_ms'] for r in values],marker='o',label=family)
        ax.set(xlabel='Initial execution budget including prelude (ms)',ylabel='Estimated full-chunk token rate (tokens/ms)',title=f'P≈{p}, N={n} — MODELED, not SLO')
        ax.grid(alpha=.25)
    axes[1].legend(fontsize=8)
    fig.suptitle('Offline coarse-context cost envelope. Decode, fairness, queue rollouts are NOT modeled.')
    fig.savefig(ROOT/'figures/execution_budget_envelope.png',dpi=170);fig.savefig(ROOT/'figures/execution_budget_envelope.svg');plt.close(fig)
    print(json.dumps(dict(candidates=len(candidates),budget_cells=len(output),phase4_status='BLOCKED' if fail else 'CONDITIONAL'),indent=2))

if __name__=='__main__':
    main()
