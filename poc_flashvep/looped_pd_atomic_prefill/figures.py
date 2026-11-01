"""Plot actual measured cohorts. Distinguish host envelopes from CUDA activity."""
import collections
import csv
import statistics
from .common import *
from .analyze import serving,capacity
from .uncertainty import bootstrap
from .evidence import rows


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs=[r for r in serving() if r['eligible_clean'] or
        (r['timing_isolation_clean'] and r['fidelity_gate_blocked'] and r['model']=='qwen8')]
    figures=ROOT/'figures';figures.mkdir(exist_ok=True);manifest=[]
    groups=collections.defaultdict(list)
    for r in runs:groups[(r['model'],r['length'],r['output_cap'],r['phase'])].append(r)
    charts=[('a_qps_ttft','TTFT_ms_p95','Request TTFT p95 (ms)'),('b_qps_tpot','request_TPOT_ms_p95','Request-average TPOT p95 (ms)'),
        ('c_qps_total_throughput','total_tokens_per_s','Finite-cohort total tokens/s'),('c_qps_output_throughput','output_tokens_per_s','Finite-cohort output tokens/s'),
        ('slo_attainment','SLO_attainment','Request joint-SLO attainment'),('slo_goodput','SLO_goodput','Finite-cohort SLO goodput (requests/s)')]
    for key,group in groups.items():
        model,length,cap,phase=key;prefix=f'{model}_l{length}_o{cap}_{phase}'
        for suffix,metric,label in charts:
            fig,ax=plt.subplots(figsize=(6.4,4.2));plotted=False
            methods=sorted({(r['arm'],r['chunk'],r['transport_control']) for r in group})
            for arm,chunk,transport in methods:
                selected=[r for r in group if r['arm']==arm and r['chunk']==chunk and r['transport_control']==transport and isinstance(r.get(metric),(int,float))]
                points=collections.defaultdict(list)
                for r in selected:points[r['target_QPS']].append(r[metric])
                if not points:continue
                x=sorted(points);y=[statistics.mean(points[q]) for q in x];bounds=[bootstrap(points[q]) for q in x]
                lo=[b[0] if b[0] is not None else v for b,v in zip(bounds,y)];hi=[b[1] if b[1] is not None else v for b,v in zip(bounds,y)]
                name=('Colocated DP2 Mixed' if arm=='DP_Mixed' else 'Colocated DP2' if arm.startswith('DP') else 'Static PD')+(' Full' if chunk==length else ' C'+str(chunk))
                if transport.startswith('INFLIGHT_'):name+=' wire='+str(int(transport.split('_')[1])//(1024**2))+'MiB'
                ax.plot(x,y,'o--' if phase=='qps_pilot' else 'o-',label=name);ax.fill_between(x,lo,hi,alpha=.15);plotted=True
            if not plotted:plt.close(fig);continue
            ax.set_xlabel('Offered target QPS');ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend(fontsize=8)
            qualification='NUMERICAL GATE FAILED; diagnostic timings only' if any(r['fidelity_gate_blocked'] for r in group) else 'N24 smoke only' if phase=='qps_pilot' else 'Finite measured cohorts; no infinite-capacity claim'
            ax.set_title(f'{model}: input {length}, output {cap}\n'+qualification,fontsize=10)
            fig.tight_layout();path=figures/(prefix+'_'+suffix+'.png');fig.savefig(path,dpi=170);plt.close(fig)
            manifest.append(dict(file=path.name,metric=metric,cell=key,source_runs=[r['name'] for r in group],
                all_settings_numerically_qualified=not any(r['fidelity_gate_blocked'] for r in group),
                uncertainty='95% bootstrap of independent-start metric means when at least two starts exist. No band otherwise.'))
        fig,ax=plt.subplots(figsize=(6.4,4.2))
        pd=[r for r in group if r['arm']=='PD_G1' and r['chunk']==length]
        for role,transport in ((role,transport) for role in ('P','D') for transport in sorted({r['transport_control'] for r in pd})):
            points=collections.defaultdict(list)
            for r in pd:
                if r['transport_control']==transport and r.get(role+'_NVML_util_mean') is not None:points[r['target_QPS']].append(r[role+'_NVML_util_mean'])
            if points:
                x=sorted(points);ax.plot(x,[statistics.mean(points[q]) for q in x],'o-',label=role+' '+transport+' NVML activity')
        if ax.lines:
            ax.set(xlabel='Offered target QPS',ylabel='NVML utilization (%)',ylim=(0,102),title=f'{model}: {length}/{cap}\nNVML activity is not SM occupancy or spare compute')
            ax.legend();ax.grid(alpha=.25);fig.tight_layout();path=figures/(prefix+'_d_nvml_activity.png');fig.savefig(path,dpi=170)
            manifest.append(dict(file=path.name,scope='Arrival-window NVML averaging samples only. Not kernel busy or SM occupancy.'))
        plt.close(fig)
    # Actual host phase envelopes are separate from device activity.
    for config_path in (ROOT/'configs').glob('profile_*final*.json'):
        cfg=json.loads(config_path.read_text());folder=ROOT/'raw'/cfg['name'];values=rows(cfg['name'])
        if not values or not (folder/'closed_stats.json').exists():continue
        window=json.loads((folder/'window.json').read_text());zero=window['start'];central=.2*(window['last_arrival']-zero)
        picked=sorted((r for r in values if r['arrival']-zero>=central),key=lambda r:r['arrival'])[:10]
        metrics={r['request_id']:r for r in json.loads((folder/'closed_stats.json').read_text()).get('metrics',[])}
        fig,axes=plt.subplots(3,1,figsize=(10,6),sharex=True,gridspec_kw={'height_ratios':[4,1,1]})
        for index,r in enumerate(picked):
            m=metrics.get('r0_q'+str(r['sample_id']))
            if not m:continue
            t=m['timings'];created=m['created_monotonic'];p0=created+t['prefill_started']-zero;p1=created+t['prefill_compute_completed']-zero
            axes[0].broken_barh([(p0,p1-p0)],(index-.3,.25),facecolors='tab:green')
            axes[0].broken_barh([(r['first_token']-zero,r['finish']-r['first_token'])],(index+.05,.25),facecolors='tab:blue')
            axes[0].plot(r['arrival']-zero,index,'k.',markersize=4)
        axes[0].set_yticks(range(len(picked)),[str(r['sample_id']) for r in picked]);axes[0].set_ylabel('Request ID')
        axes[0].set_title('Measured host envelopes: P green, D generation blue, arrival black.\nThese envelopes are not GPU kernel execution intervals.',fontsize=10)
        path=ROOT/'profiles'/(cfg['name']+'_kernel_intervals.csv')
        intervals=list(csv.DictReader(path.open())) if path.exists() else []
        for role,ax in zip(('prefill','decode'),axes[1:]):
            data=[(float(r['start_s']),float(r['end_s'])-float(r['start_s'])) for r in intervals if r['role']==role]
            if data:ax.broken_barh(data,(0,1),facecolors='tab:orange',rasterized=True)
            ax.set_yticks([]);ax.set_ylabel(role,fontsize=8)
            ax.text(.01,.78,'Recorded kernels only. Graph internals may be absent. No idle inference.',transform=ax.transAxes,fontsize=7)
        axes[-1].set_xlabel('Seconds from measured-window start')
        if picked:axes[-1].set_xlim(central,max(r['finish']-zero for r in picked))
        fig.tight_layout();path=figures/(cfg['model']+'_e_actual_request_timeline.png');fig.savefig(path,dpi=170);plt.close(fig)
        manifest.append(dict(file=path.name,source=cfg['name'],evidence='MEASURED_PROFILE_AND_HOST_PHASES',scope='No fabricated device idle. Host envelopes and recorded device intervals are separate.'))
    write(figures/'figure_manifest.json',manifest);print('FIGURES',len(manifest));return manifest


if __name__=='__main__':main()
