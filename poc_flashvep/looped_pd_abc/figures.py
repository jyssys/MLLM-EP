"""Editable SVG figures from completed measurements. Never plot model costs as live latency."""
import csv
from collections import defaultdict
import numpy as np
from .common import ROOT
from .analyze import requests,save_csv
from .frozen_runs import preferred,matches
from .scope import active_models

def make(summary):
 summary=[row for row in summary if str(row.get('clean_serving_eligible')).lower()=='true' and row['model'] in active_models()]
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams['svg.fonttype']='none'
 provenance=[]
 def selected(model,label,length,width):
  names={preferred(f'{model}_{label}_P{length}_C{width}_s{i}') for i in range(3)}
  return [row for row in summary if row['name'] in names]
 def track(figure,rows):
  for row in rows:
   provenance.append(dict(figure=figure,run=row['name'],physical_GPUs=row['gpus'],N=row['requests'],
    output_cap=row.get('output_cap','NOT_AVAILABLE'),frozen_runtime=matches(ROOT/'raw'/row['name']),
    evidence='MEASURED_COMPLETED_RUN',timing='Clean harness wall time, not profiler timing'))
 colors={'DP':'#333333','DP_Mixed':'#777777','PD':'#b35806','PD_G1':'#9970ab','A':'#2166ac','B':'#1a9850'}
 labels=tuple(colors)

 # Recurrence split tuning uses a different cap and must not become an accepted gain figure.
 fig,axes=plt.subplots(2,len(active_models()),figsize=(13,7))
 for col,model in enumerate(active_models()):
  for row_index,length in enumerate((512,2048)):
   ax=axes[row_index,col]
   rows=[r for r in summary if r['name'].startswith(model+'_A_sweep_') and f'_P{length}_C4_' in r['name']]
   values=[r for r in rows if '_DP_' not in r['name'] and '_rR_G1_' not in r['name']]
   values.sort(key=lambda r:float(r['r_boundary']))
   if values:ax.plot([float(r['r_boundary']) for r in values],[float(r['e2e_ms_p95']) for r in values],'o-',label='Split sweep / atomic endpoint')
   for control,style in (('_DP_','--'),('_rR_G1_',':')):
    found=[r for r in rows if control in r['name']]
    if found:ax.axhline(float(found[0]['e2e_ms_p95']),linestyle=style,color='#777777',label='DP2' if control=='_DP_' else 'Full-prefill G1 PD')
   ax.set_title(f'{model}, P{length}, C4');ax.set_xlabel('P-owned recurrent loops r');ax.set_ylabel('Request p95 E2E (ms)');ax.grid(alpha=.2)
   if rows:track('A_cut_sensitivity.svg',rows)
   else:ax.text(.5,.5,'NOT_MEASURED',ha='center',transform=ax.transAxes)
 axes[0,-1].legend(fontsize=7)
 fig.suptitle('MEASURED single-start tuning; N16 per point; natural outputs capped at 8; not accepted gain evidence')
 fig.tight_layout();fig.savefig(ROOT/'figures/A_cut_sensitivity.svg');plt.close(fig)

 fig,axes=plt.subplots(1,len(active_models()),figsize=(13,4.1))
 for ax,model in zip(axes,active_models()):
  for label in labels:
   for (length,width),marker in zip(((128,1),(512,4),(2048,8)),('o','s','^')):
    rows=selected(model,label,length,width)
    if not rows:continue
    ax.scatter(np.median([float(r['ttft_ms_p95']) for r in rows]),np.median([float(r['output_tokens_s']) for r in rows]),
     color=colors[label],marker=marker,s=35,label=label if length==128 else None)
    track('throughput_vs_TTFT.svg',rows)
  ax.set_title(model);ax.set_xlabel('Median per-start p95 TTFT (ms)');ax.set_ylabel('Median output tokens/s');ax.grid(alpha=.2)
 if axes[-1].get_legend_handles_labels()[0]:axes[-1].legend(fontsize=7)
 fig.suptitle('MEASURED two-GPU serving, cap32; circle=P128/C1, square=P512/C4, triangle=P2048/C8')
 fig.tight_layout();fig.savefig(ROOT/'figures/throughput_vs_TTFT.svg');plt.close(fig)

 fig,axes=plt.subplots(1,len(active_models()),figsize=(13,4.1))
 for ax,model in zip(axes,active_models()):
  for label in labels:
   rows=selected(model,label,512,4);times=[]
   for row in rows:times.extend(r['e2e_ms'] for r in requests(ROOT/'raw'/row['name']))
   if not times:continue
   ordered=np.sort(times);tail=(len(times)-np.arange(len(times)))/len(times)
   ax.step(ordered,tail,where='post',color=colors[label],label=f'{label}, N={len(times)}, starts={len(rows)}')
   track('E2E_tail_P512_C4.svg',rows)
  ax.set_title(model);ax.set_xlabel('Request E2E (ms)');ax.set_ylabel('Empirical fraction at or above latency');ax.grid(alpha=.2)
  if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=6)
 fig.suptitle('MEASURED bounded natural generation, P512/C4, cap32; preferred completed starts; no accepted claim without fidelity')
 fig.tight_layout();fig.savefig(ROOT/'figures/E2E_tail_P512_C4.svg');plt.close(fig)

 phase_path=ROOT/'e2e_phase_breakdown.csv';phase_groups=defaultdict(list)
 if phase_path.exists():
  with phase_path.open() as stream:
   for row in csv.DictReader(stream):
    if row['phase_order_valid']=='True':phase_groups[row['name']].append(float(row['uncovered_handoff_ms']))
 fig,axes=plt.subplots(1,len(active_models()),figsize=(13,4.1))
 for ax,model in zip(axes,active_models()):
  for label in ('PD','PD_G1','B'):
   rows=selected(model,label,512,4);values=[]
   for row in rows:
    if not row.get('measured_transfer_count') or not phase_groups[row['name']]:continue
    values.append((float(row['measured_transfer_count'])/float(row['requests']),np.mean(phase_groups[row['name']])))
   if values:
    x,y=np.mean(values,axis=0);ax.scatter(x,y,s=45,color=colors[label]);ax.annotate(label,(x,y),xytext=(3,3),textcoords='offset points')
    track('B_transfer_granularity.svg',rows)
  ax.set_title(model);ax.set_xlabel('Observed completed NIXL transfers/request');ax.set_ylabel('Mean host uncovered handoff (ms)');ax.grid(alpha=.2)
 fig.suptitle('MEASURED P512/C4/cap32; PD_G1 is the matched overlapped control; host handoff is not pure DMA time')
 fig.tight_layout();fig.savefig(ROOT/'figures/B_transfer_granularity.svg');plt.close(fig)
 save_csv(ROOT/'figure_provenance.csv',provenance)

if __name__=='__main__':
 with (ROOT/'results_raw.csv').open() as stream:make(list(csv.DictReader(stream)))
