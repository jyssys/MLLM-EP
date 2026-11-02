"""Plot actual measured cohorts. Distinguish host envelopes from CUDA activity."""
import collections
import csv
import statistics
import re
from .common import *
from .analyze import serving,capacity
from .uncertainty import bootstrap
from .evidence import rows,records


def short_label(row):
    arm=row['arm'];transport=row['transport_control']
    base='DP2 Mixed' if arm=='DP_Mixed' else 'DP2' if arm.startswith('DP') else 'PD'
    chunk='Full' if row['chunk']==row['length'] else 'C'+str(row['chunk'])
    label=base+' '+('chunk cap ' if arm.startswith('DP') else '')+chunk
    if transport.startswith('INFLIGHT_'):
        credit=int(transport.split('_')[1]);label+='; wire '+f'{credit/(1024**3):g} GiB'
    if not arm.startswith('DP'):
        label+='; P'+str(row['P_slots'])+' / D-budget '+str(row['D_token_budget'])
        drain=re.search(r'_DRAIN(\d+)',transport)
        if drain:label+='; drain '+drain.group(1)
    else:
        budget=re.search(r'_B(\d+)',transport)
        if budget:label+='; budget '+budget.group(1)
    pool=row.get('D_KV_blocks')
    if isinstance(pool,int):label+='; KV '+str(pool)
    return label


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs=[r for r in serving() if r['eligible_clean'] or
        (r['timing_isolation_clean'] and r['fidelity_gate_blocked'] and r['model']=='qwen8')]
    figures=ROOT/'figures';figures.mkdir(exist_ok=True);manifest=[]
    groups=collections.defaultdict(list)
    for r in runs:groups[(r['model'],r['length'],r['output_cap'],r['phase'],r.get('execution_family','ORIGINAL'))].append(r)
    charts=[('a_qps_ttft','TTFT_ms_p95','Start-mean request TTFT p95 (ms)'),('b_qps_tpot','request_TPOT_ms_p95','Start-mean request-average TPOT p95 (ms)'),
        ('c_qps_total_throughput','total_tokens_per_s','Finite-cohort total tokens/s'),('c_qps_output_throughput','output_tokens_per_s','Finite-cohort output tokens/s'),
        ('slo_attainment','SLO_attainment','Request joint-SLO attainment'),('slo_goodput','SLO_goodput','Finite-cohort SLO goodput (requests/s)')]
    for key,group in groups.items():
        model,length,cap,phase,family=key;prefix=f'{model}_l{length}_o{cap}_{phase}'+('' if family=='ORIGINAL' else '_'+family)
        for suffix,metric,label in charts:
            fig,ax=plt.subplots(figsize=(10.5,4.6));plotted=False
            methods=sorted({(r['arm'],r['chunk'],r['transport_control']) for r in group})
            for arm,chunk,transport in methods:
                selected=[r for r in group if r['arm']==arm and r['chunk']==chunk and r['transport_control']==transport and isinstance(r.get(metric),(int,float))]
                points=collections.defaultdict(list)
                for r in selected:points[r['target_QPS']].append(r[metric])
                if not points:continue
                x=sorted(points);y=[statistics.mean(points[q]) for q in x];bounds=[bootstrap(points[q]) for q in x]
                lo=[b[0] if b[0] is not None else v for b,v in zip(bounds,y)];hi=[b[1] if b[1] is not None else v for b,v in zip(bounds,y)]
                name=short_label(selected[0])
                line,=ax.plot(x,y,'o--' if phase=='qps_pilot' else 'o-',label=name)
                ax.fill_between(x,lo,hi,alpha=.15)
                if len(x)==1 and len(points[x[0]])>1:
                    ax.errorbar(x,y,yerr=[[max(v-a,0) for v,a in zip(y,lo)],[max(b-v,0) for v,b in zip(y,hi)]],
                        fmt='none',capsize=4,color=line.get_color())
                plotted=True
            if not plotted:plt.close(fig);continue
            ax.set_xlabel('Offered target QPS');ax.set_ylabel(label);ax.grid(alpha=.25)
            ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.02,1.))
            qualification='NUMERICAL GATE FAILED; diagnostic timings only' if any(r['fidelity_gate_blocked'] for r in group) else f'Exploratory cohorts: N{min(r["offered"] for r in group)}–N{max(r["offered"] for r in group)}' if phase=='qps_pilot' else 'Finite measured cohorts; no infinite-capacity claim'
            ax.set_title(f'{model}: input {length}, output {cap}\n'+qualification,fontsize=10)
            fig.tight_layout();path=figures/(prefix+'_'+suffix+'.png');fig.savefig(path,dpi=170);plt.close(fig)
            manifest.append(dict(file=path.name,metric=metric,cell=key,source_runs=[r['name'] for r in group],
                all_settings_numerically_qualified=not any(r['fidelity_gate_blocked'] for r in group),
                uncertainty='95% bootstrap of independent-start metric means when at least two starts exist. No band otherwise.'))
        fig,ax=plt.subplots(figsize=(10.5,4.6))
        pd=[r for r in group if r['arm']=='PD_G1' and r['chunk']==length]
        for role,transport in ((role,transport) for role in ('P','D') for transport in sorted({r['transport_control'] for r in pd})):
            points=collections.defaultdict(list)
            for r in pd:
                if r['transport_control']==transport and r.get(role+'_NVML_util_mean') is not None:points[r['target_QPS']].append(r[role+'_NVML_util_mean'])
            if points:
                setting=next(r for r in pd if r['transport_control']==transport)
                x=sorted(points);ax.plot(x,[statistics.mean(points[q]) for q in x],'o-',label=role+' '+short_label(setting))
        if ax.lines:
            ax.set(xlabel='Offered target QPS',ylabel='NVML utilization (%)',ylim=(0,102),title=f'{model}: {length}/{cap}\nNVML activity is not SM occupancy or spare compute')
            ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.02,1.));ax.grid(alpha=.25);fig.tight_layout();path=figures/(prefix+'_d_nvml_activity.png');fig.savefig(path,dpi=170)
            manifest.append(dict(file=path.name,scope='Arrival-window NVML averaging samples only. Not kernel busy or SM occupancy.'))
        plt.close(fig)
    # Actual host phase envelopes are separate from device activity.
    for config_path in (ROOT/'configs').glob('*.json'):
        cfg=json.loads(config_path.read_text())
        if not isinstance(cfg,dict) or not cfg.get('name') or not (cfg.get('nsys') or cfg.get('nsys_P_only')) or cfg.get('phase')!='profile_control':continue
        folder=ROOT/'raw'/cfg['name'];values=rows(cfg['name'])
        if not values or not (folder/'window.json').exists():continue
        window=json.loads((folder/'window.json').read_text());zero=window['start'];central=.2*(window['last_arrival']-zero)
        picked=sorted((r for r in values if r['arrival']-zero>=central),key=lambda r:r['arrival'])[:10]
        metrics={r['request_id']:r for r in json.loads((folder/'closed_stats.json').read_text()).get('metrics',[])} if (folder/'closed_stats.json').exists() else {}
        handoffs={r['request_id']:r['time'] for r in records(folder/'native_handoffs.jsonl')}
        native_starts={}
        first_submissions={}
        for path in folder.glob('host_execution_*.json'):
            for event in json.loads(path.read_text()).get('frontiers',[]):
                if event['event']=='P_FIRST_SUBMISSION':first_submissions[event['request_id']]=event['time']
        for path in folder.glob('native_prefill_*.json'):
            for span in json.loads(path.read_text())['prompt_spans']:
                rid=span['request_id'].split('-',1)[0];native_starts[rid]=min(native_starts.get(rid,float('inf')),span['time'])
        fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,gridspec_kw={'height_ratios':[4,1,1]})
        for index,r in enumerate(picked):
            m=metrics.get('r0_q'+str(r['sample_id']))
            if m:
                t=m['timings'];created=m['created_monotonic'];received=created+t['prefill_started']-zero
                p0=first_submissions.get(m['transfer_id'],created+t['prefill_started'])-zero
                p1=created+t['prefill_compute_completed']-zero
                if p0>received:axes[0].broken_barh([(received,p0-received)],(index-.3,.25),facecolors='lightgrey')
                for key,marker,color in [('admitted','^','grey'),('d_received','D','tab:orange'),('d_activated','*','tab:red')]:
                    if key in t:axes[0].plot(created+t[key]-zero,index,marker=marker,color=color,markersize=4)
            else:
                rid='q'+str(r['sample_id'])
                if rid not in native_starts or rid not in handoffs:continue
                p0=native_starts[rid]-zero;p1=handoffs[rid]-zero
                axes[0].plot(r['admission']-zero,index,marker='^',color='grey',markersize=4)
                axes[0].plot(p1,index,marker='D',color='tab:orange',markersize=4)
            axes[0].broken_barh([(p0,p1-p0)],(index-.3,.25),facecolors='tab:green')
            axes[0].broken_barh([(r['first_token']-zero,r['finish']-r['first_token'])],(index+.05,.25),facecolors='tab:blue')
            axes[0].plot(r['arrival']-zero,index,'k.',markersize=4)
        axes[0].set_yticks(range(len(picked)),[str(r['sample_id']) for r in picked]);axes[0].set_ylabel('Request ID')
        handoff_label='AR handoff submission, not D receipt' if cfg['model']=='qwen8' else 'native valid-state receipt'
        axes[0].set_title('Host envelopes: P green, D generation blue, P wait grey. Not GPU kernel intervals.\n'
            'Arrival dot; admission triangle; handoff diamond; D activation star.\nOrange diamond: '+handoff_label+'.',fontsize=9)
        path=ROOT/'profiles'/(cfg['name']+'_kernel_intervals.csv')
        right=max((r['finish']-zero for r in picked),default=central+1)
        intervals=[]
        if path.exists():
            with path.open() as stream:
                for r in csv.DictReader(stream):
                    if float(r['end_s'])>central and float(r['start_s'])<right:intervals.append(r)
        for role,ax in zip(('prefill','decode'),axes[1:]):
            data=[(float(r['start_s']),float(r['end_s'])-float(r['start_s'])) for r in intervals if r['role']==role]
            merged=[]
            for start,duration in sorted(data):
                end=start+duration
                if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
                else:merged.append([start,end])
            if merged:ax.broken_barh([(a,b-a) for a,b in merged],(0,1),facecolors='tab:orange',rasterized=True)
            ax.set_yticks([]);ax.set_ylabel(role,fontsize=8)
            label='D excluded from this P-only trace. No idle inference.' if role=='decode' and cfg.get('nsys_P_only') else 'Recorded kernel union only. Graph internals may be absent. No idle inference.'
            ax.text(.01,.78,label,transform=ax.transAxes,fontsize=7)
        axes[-1].set_xlabel('Seconds from measured-window start')
        if picked:axes[-1].set_xlim(central,max(r['finish']-zero for r in picked))
        fig.tight_layout();path=figures/(cfg['name']+'_e_actual_request_timeline.png');fig.savefig(path,dpi=170);plt.close(fig)
        manifest.append(dict(file=path.name,source=cfg['name'],evidence='MEASURED_PROFILE_AND_HOST_PHASES',scope='No fabricated device idle. Host envelopes and recorded device intervals are separate.'))
    write(figures/'figure_manifest.json',manifest);print('FIGURES',len(manifest));return manifest


if __name__=='__main__':main()
