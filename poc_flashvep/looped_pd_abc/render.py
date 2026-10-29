"""Render evidence tables and figures. Unreviewed decisions remain NOT_MEASURED."""
import csv
from datetime import datetime,timezone
import json
import numpy as np
from .common import ROOT
from .analyze import save_csv

def read(name):
 path=ROOT/name
 return list(csv.DictReader(path.open())) if path.exists() else []

def reviewed_decisions():
 path=ROOT/'configs/reviewed_decisions.json'
 if not path.exists():return {}
 value=json.loads(path.read_text())
 if value.get('review_complete') is not True:raise ValueError('Decision file requires an explicit completed scientific review')
 allowed={'GO','CONDITIONAL','NO_GO_PERFORMANCE','NO_GO_NOVELTY','NO_GO_CORRECTNESS',
  'BLOCKED_IMPLEMENTATION','BLOCKED_RESOURCE','NOT_MEASURED'}
 for track in ('A','B','C'):
  overall=value.get(track,{}).get('overall',{})
  if overall.get('status') not in allowed or not overall.get('reason'):raise ValueError('Missing independent overall decision: '+track)
  for model in ('ouro','huginn','dory'):
   item=value.get(track,{}).get(model,{})
   if item.get('status') not in allowed or not item.get('reason'):raise ValueError('Missing model-specific decision: '+track+'/'+model)
   for count in ('2','4','8'):
    if item.get('scale_status',{}).get(count) not in allowed:raise ValueError('Missing scale-specific decision: '+track+'/'+model+'/'+count)
 return value

def table(headers,rows):
 def cell(value):
  if isinstance(value,float):return f'{value:.3f}'
  return str(value).replace('|','/').replace('\n',' ')
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
  *['| '+' | '.join(cell(x) for x in row)+' |' for row in rows]])+'\n'

def document(name,title,text):
 state=ROOT/'state/main_campaign.json'
 status=json.loads(state.read_text()).get('status') if state.exists() else 'NOT_STARTED'
 banner='Reviewed decision file present.' if reviewed_decisions() else 'DRAFT: final scientific review is pending. No GO claim is accepted.'
 if (ROOT/'configs/evidence_protocol_epoch.json').exists():
  banner+='\n\nTiming correction: legacy foreground-I/O runs are not valid clean method comparisons. Read `timing_correction.md`. Accepted comparisons require the registered nonblocking protocol.'
 stamp=datetime.now(timezone.utc).isoformat()
 (ROOT/name).write_text(f'# {title}\n\n{banner}\n\nCampaign status: `{status}`. Evidence snapshot: {stamp}.\n\n'+text)

def plots(summary,comparisons):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams['svg.fonttype']='none'
 from .frozen_runs import preferred
 def selected(prefix):
  names={preferred(prefix+str(i)) for i in range(3)}
  return [r for r in summary if r['name'] in names and str(r.get('clean_serving_eligible')).lower()=='true']
 colors={'DP':'#595959','DP_Mixed':'#171717','PD':'#929292','PD_G1':'#8c6d31','PD_A_control':'#c7a76c','PD_A_coalesced':'#a6761d','A':'#2166ac','B':'#1a9850'}
 for track in ('A','B'):
  fig,axes=plt.subplots(1,3,figsize=(13,3.6))
  for ax,model in zip(axes,('ouro','huginn','dory')):
   labels=('DP','DP_Mixed','PD','PD_G1','PD_A_control')+(('PD_A_coalesced',) if model=='ouro' else ())+(track,)
   for label in labels:
    xs=[];ys=[];starts=[]
    for x,(length,width) in enumerate(((128,1),(512,4),(2048,8))):
     rows=selected(f'{model}_{label}_P{length}_C{width}_s')
     if rows:xs.append(x);ys.append(float(np.median([float(r['e2e_ms_p95']) for r in rows])));starts.append(len(rows))
    if xs:ax.plot(xs,ys,'o-',label=label,color=colors[label],linewidth=1.5)
   ax.set_title(model);ax.set_xticks(range(3),['P128/C1','P512/C4','P2048/C8']);ax.set_ylabel('Median per-start request p95 E2E (ms)')
   ax.grid(alpha=.2)
  if axes[-1].get_legend_handles_labels()[0]:axes[-1].legend(fontsize=7)
  fig.suptitle(f'Track {track}: isolated two-GPU serving; cap 32; preferred completed starts; drafts may lack three frozen starts')
  fig.tight_layout();fig.savefig(ROOT/f'figures/track_{track}.svg');plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,3.6))
 for ax,model in zip(axes,('ouro','huginn','dory')):
  labels=[];times=[]
  for label in ('C_DP4','C_DP4_Mixed','C_PD1','C_PD3','C_upstream','C_queue','C_token','C_service','C_loop'):
   rows=selected(f'{model}_{label}_P0_C8_s')
   if rows:labels.append(label.removeprefix('C_'));times.append(float(np.median([float(r['e2e_ms_p95']) for r in rows])))
  if times:ax.bar(range(len(times)),times,color=['#2166ac' if x=='loop' else '#777777' for x in labels])
  else:ax.text(.5,.5,'NOT_MEASURED',ha='center',transform=ax.transAxes)
  ax.set_title(model);ax.set_xticks(range(len(labels)),labels,rotation=50,ha='right');ax.set_ylabel('Median per-start request p95 E2E (ms)')
 fig.suptitle('Track C: actual isolated four-GPU natural-prompt serving; C8; cap 64; completed starts only')
 fig.tight_layout();fig.savefig(ROOT/'figures/track_C.svg');plt.close(fig)
 refs=read('reference_agreement.csv');fig,ax=plt.subplots(figsize=(6,3.6))
 if refs:
  names=[r['model']+('\nexplicit R4' if 'explicit_last_R4' in r['reference'] else '\ndefault reference') for r in refs]
  values=[float(r['agreement_percent']) for r in refs]
  ax.bar(range(len(names)),values,color='#b35806');ax.set_xticks(range(len(names)),names)
  ax.set_ylim(0,105);ax.set_ylabel('Exact sequence agreement (%)')
  for i,r in enumerate(refs):ax.text(i,values[i]+1,f"{r['exact']}/{r['N']}",ha='center')
 ax.set_title('Native RLT / official reference, cap 32\nMEASURED output fidelity, not answer accuracy')
 fig.tight_layout();fig.savefig(ROOT/'figures/reference_correctness.svg');plt.close(fig)
 from .critical_path_analysis import PHASES
 shares=read('e2e_phase_percent_summary.csv');fig,ax=plt.subplots(figsize=(12,5.5))
 names=[];vectors=[]
 for model in ('ouro','huginn','dory'):
  for label in ('PD','PD_G1','A','B'):
   names_in_plot={r['name'] for r in selected(f'{model}_{label}_P512_C4_s')}
   totals=[]
   for phase in PHASES:
    totals.append(sum(float(r['N'])*float(r['phase_mean_ms']) for r in shares
     if r['run'] in names_in_plot and r['phase']==phase))
   if sum(totals)>0:names.append(model+' '+label);vectors.append(np.asarray(totals)/sum(totals)*100)
 if vectors:
  values=np.asarray(vectors);left=np.zeros(len(names))
  for i,phase in enumerate(PHASES):
   ax.barh(range(len(names)),values[:,i],left=left,label=phase.replace('_',' '));left+=values[:,i]
  ax.set_yticks(range(len(names)),names);ax.set_xlim(0,100)
 else:ax.text(.5,.5,'NOT_MEASURED',ha='center',transform=ax.transAxes)
 ax.set_xlabel('Share of summed request E2E time (%) — not GPU utilization')
 ax.set_title('Measured host critical-path partition; P512/C4; cap 32; preferred completed starts')
 ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(1,1))
 fig.tight_layout();fig.savefig(ROOT/'figures/host_E2E_partition.svg');plt.close(fig)

def main():
 summary=read('results_raw.csv');comparisons=read('paired_comparisons.csv')
 decisions=reviewed_decisions()
 scope=('These are actual GPU measurements from completed runs.\n'
  'Unfinished and failed runs do not contribute latency values.\n'
  'Natural generation permits EOS and preserves the declared output cap.\n'
  'Cap termination is not proof of answer completeness.\n'
  'A lower latency with different output IDs is not an accepted lossless speedup.\n\n')
 limits=('## Limits\n\nOfficial-reference sequence agreement is incomplete for all three models.\n'
  'Full GSM8K test accuracy and HumanEval accuracy are `NOT_MEASURED`.\n'
  'A closed-loop cohort does not establish sustainable open-loop capacity.\n'
  'No global queue simulator or infinite-bandwidth serving oracle is validated.\n'
  'GPU kernel times and overlapping transfer durations must not be added to E2E.\n'
  'Short profiled runs are separate from clean serving runs.\n')
 headers=['Run','GPU IDs','P:D / replicas','N','Output cap','TTFT p95 ms','TPOT mean ms','E2E p95 ms','Requests/s','Peak Torch allocation bytes','Sampled cohort NVML HBM bytes','Timing evidence']
 def master(rows):
  values=[]
  for r in rows:
   placement=f"{r['P']}:{r.get('D','?')}" if r['method']=='pd' else f"{r.get('colocated_replicas','?')} colocated"
   values.append([r['name'],r['gpus'],placement,r['requests'],r.get('output_cap','NOT_AVAILABLE'),
    float(r['ttft_ms_p95']),r['tpot_ms_mean'],float(r['e2e_ms_p95']),float(r['requests_s']),r.get('peak_HBM_bytes_max','NOT_MEASURED'),r.get('NVML_sampled_cohort_peak_HBM_bytes_max','NOT_MEASURED'),r.get('evidence','NOT_AVAILABLE')])
  return table(headers,values) if values else 'NOT_MEASURED.\n'
 scale=[r for r in summary if '_scale' in r['name']]
 document('04_baseline_2_4_8gpu.md','Equal-budget baseline measurements',scope+master(scale)+'\n'+limits)
 comparison_headers=['Model / regime','Baseline','Candidate','Starts / frozen pairs','Exact outputs / N','Baseline p95 E2E ms','Candidate p95 E2E ms','Speedup','Paired 95% CI']
 def matched(rows):
  result=[]
  for r in rows:
   interval=json.loads(r['p95_E2E_speedup_ci95'])
   result.append([f"{r['model']} / {r['kind']} / P{r['input_tokens']} C{r['concurrency']}",r['baseline'],r['candidate'],
    f"{r['starts']} / {r.get('matched_frozen_runtime_starts','NOT_AVAILABLE')}",f"{r['exact_outputs']}/{r['N']}",float(r['baseline_p95_E2E_ms']),float(r['candidate_p95_E2E_ms']),
    f"{float(r['p95_E2E_speedup']):.3f}×",f"{interval[0]:.3f}–{interval[1]:.3f}×"])
  return table(comparison_headers,result) if result else 'NOT_MEASURED. Repeated matched comparisons are pending.\n'
 rows_board=[];scale_board=[];overall_board=[]
 for track in ('A','B','C'):
  data=[r for r in comparisons if (r['candidate']=='C_loop' if track=='C' else r['candidate'] in (track,'L128_'+track)
   or track=='B' and r['candidate']=='Bpace_frontier_confirm')]
  text=scope+matched(data)+'\n## Mechanism and controls\n\n'
  if track=='A':
   text+=('A transfers full-prompt recurrent state at an intermediate cut.\n'
    'The destination completes prefill and then executes full-depth decode.\n'
    'The matched A control uses the same whole-prompt chunk and staged execution.\n'
    'Generic logical-depth migration is a novelty counterexample.\n'
    'The cut sweep is separate tuning evidence, not an accepted single-start gain.\n')
  elif track=='B':
   text+=('B changes transfer timing, not compute ownership or prescribed depth.\n'
    'The main matched control retains upstream chunk-transfer overlap and staged G1.\n'
    'The Ouro control also includes generic descriptor coalescing.\n'
    'Boundary-layer controls exclude invalid unused planes in both arms.\n'
    'Generic immutable layerwise streaming remains a novelty counterexample.\n')
   text+=('The conditional Dory control sweep changes only generic NIXL chunk and inflight-byte limits.\n'
    'The confirmation compares matched transport budgets. These knobs are not a loop-specific contribution.\n'
    'SLO bootstrap results appear in `paired_SLO_comparisons.csv`.\n')
  else:
   text+=('C changes initial placement only. A and B remain disabled.\n'
    'The main feature ablation compares loop-state placement with calibrated service-time placement.\n'
    'Both policies receive the same legal progress telemetry.\n'
    'C2 migration and dynamic role reassignment are NOT_MEASURED.\n'
    'A one-prefill/one-decode pool has no initial routing choice.\n')
  document(f'06{track}_implementation_and_results.md',f'Track {track}: measured implementation results',text+'\n'+limits)
  decision_rows=[]
  for model in ('ouro','huginn','dory'):
   item=decisions.get(track,{}).get(model,{})
   status=item.get('status','NOT_MEASURED');reason=item.get('reason','Final review pending. Do not interpret partial comparisons as a final verdict.')
   decision_rows.append([model,status,reason]);rows_board.append(dict(track=track,model=model,status=status,reason=reason,evidence='REVIEWED_DECISION' if item else 'NOT_MEASURED_PENDING_REVIEW'))
   for count in (2,4,8):
    scale_board.append(dict(track=track,model=model,GPU_count=count,status=item.get('scale_status',{}).get(str(count),'NOT_MEASURED'),
     scope_note=item.get('scale_notes',{}).get(str(count),'Review pending. Consult the measurement tables for actual completed runs.'),
     evidence='REVIEWED_DECISION' if item else 'NOT_MEASURED_PENDING_REVIEW'))
  overall=decisions.get(track,{}).get('overall',{})
  overall_board.append(dict(track=track,status=overall.get('status','NOT_MEASURED'),reason=overall.get('reason','Scientific review pending.')))
  document(f'07{track}_final_decision.md',f'Track {track}: decision',
   f"Overall status: `{overall.get('status','NOT_MEASURED')}`. {overall.get('reason','Scientific review pending.')}\n\n"+
   table(['Model','Status','Reason'],decision_rows)+'\n'+matched(data)+'\n'+limits)
 save_csv(ROOT/'aggregate_decisions.csv',rows_board);save_csv(ROOT/'comparison_matrix.csv',scale_board)
 save_csv(ROOT/'overall_decisions.csv',overall_board)
 document('08_comparison_matrix.md','Independent A/B/C decision matrix',table(['Track','Model','Status','Reason'],[[r[k] for k in ('track','model','status','reason')] for r in rows_board])+'\n'+
  table(['Track','Model','GPU budget','Status','Scope'],[[r[k] for k in ('track','model','GPU_count','status','scope_note')] for r in scale_board])+'\n'+limits)
 refs=read('reference_agreement.csv')
 ref_table=table(['Model','Reference run / protocol','Exact output / N','Wilson 95% CI, %','GSM8K test','HumanEval'],[
  [r['model'],r['reference'],f"{r['exact']}/{r['N']}",r['wilson_95_percent'],'NOT_MEASURED','NOT_MEASURED'] for r in refs])
 document('09_final_summary.md','Dense Looped PD: evidence and decisions',scope+
  table(['Track','Overall status','Reason'],[[r[k] for k in ('track','status','reason')] for r in overall_board])+'\n'+ref_table+'\n'+
  table(['Track','Model','Status'],[[r[k] for k in ('track','model','status')] for r in rows_board])+'\n'+
  '## Evidence locations\n\nThe master measurement table is `results_raw.csv`.\n'
  'The repeated comparisons are `paired_comparisons.csv`.\n'
  'The frozen-grid request SLO results are `slo_results.csv`.\n'
  'The exact-transfer results are `transfer_identity.csv`.\n'
  'The full-depth submission accounting is `full_depth_work_accounting.csv`.\n'
  'The host phase partition is `e2e_phase_breakdown.csv`.\n'
  'The E2E component percentages are `e2e_phase_percent_summary.csv`.\n'
  'The frozen-timeline handoff oracle is `handoff_frozen_timeline_oracle.csv`. It is not measured speedup.\n'
  'The request outputs and source snapshots remain under `raw/`.\n\n'+limits)
 plots(summary,comparisons)
 from .figures import make as additional_figures
 additional_figures(summary)

if __name__=='__main__':main()
