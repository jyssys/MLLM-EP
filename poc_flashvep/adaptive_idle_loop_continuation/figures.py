"""Draw measured enclosures and explicit missing results. Never invent idle."""
import csv
import math
from .common import *


def main():
    initialize()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from .analyze import jsonlines
    from ..d_local_cross_depth_batching.profile_analysis import union
    from ..d_local_cross_depth_batching.analyze import pct
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'figure.dpi':150,
        'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
    manifest=[]
    def load(path):return json.loads(path.read_text())
    def csvrows(path):return list(csv.DictReader(path.open())) if path.exists() else []
    def save(fig,name,sources,scope):
        fig.tight_layout();paths=[]
        for extension in ('png','pdf'):
            path=ROOT/'figures'/(name+'.'+extension);fig.savefig(path);paths.append(str(path.relative_to(ROOT)))
        plt.close(fig)
        manifest.append(dict(name=name,files=paths,sources=[dict(path=str(p.relative_to(ROOT)),SHA256=digest(p))
            for p in dict.fromkeys(sources) if p.exists()],scope=scope))
    def missing(ax,message='NOT_MEASURED'):
        ax.text(.5,.5,message,ha='center',va='center',transform=ax.transAxes,color='#7a263a');ax.set_xticks([]);ax.set_yticks([])
    primitives=load(ROOT/'full_prompt_primitive_costs.json')
    costs={model:sorted([r for r in primitives if r['model']==model and r['remaining_loops']>0],key=lambda r:r['remaining_loops']) for model in ('ouro','huginn')}
    gaps=load(ROOT/'idle_gap_distributions.json')
    chosen={}
    for model in ('ouro','huginn'):
        candidates=[r for r in gaps if r['model']==model and not r.get('status') and
            (ROOT/'raw'/r['name']/'compute_free_upper_envelope_gaps.csv').exists()]
        candidates.sort(key=lambda r:(not r['qualified_instrumentation'],r['observation_start']))
        chosen[model]=candidates[0] if candidates else None

    # 1: One bounded serving window. Host pressure is not GPU saturation.
    fig,axes=plt.subplots(2,3,figsize=(16,7));sources=[]
    for index,model in enumerate(('ouro','huginn')):
        record=chosen[model]
        for ax in axes[index]:ax.set_title(model.capitalize())
        if record is None:
            for ax in axes[index]:missing(ax,'No eligible activity-bound trace')
            continue
        folder=ROOT/'raw'/record['name'];left,right=record['observation_start'],record['observation_end']
        pressure=csvrows(folder/'P_pressure_timeline.csv');sources.append(folder/'P_pressure_timeline.csv')
        points=[v for v in pressure if left<=float(v['time'])<=right and v['P_backlog_lower_bound']!='']
        axes[index,0].step([float(v['time'])-left for v in points],[int(v['P_backlog_lower_bound']) for v in points],where='post')
        axes[index,0].set(xlabel='Time in capture (s)',ylabel='D-credit-clear P-slot pressure\n+ admitted owner queue')
        marker=next(folder.glob('idle_observer_decode_*.json'));observer=load(marker);capture=observer['capture']
        def host(stamp):
            a,b=capture['start'],capture['stop']
            return a['monotonic_s']+(stamp-a['device_clock_ns'])/(b['device_clock_ns']-a['device_clock_ns'])*(b['monotonic_s']-a['monotonic_s'])
        trace=folder/capture['file'];sources.extend((marker,trace))
        mapping=load(folder/'worker_map.json');sources.append(folder/'worker_map.json')
        device=next(v['visible_device'] for v in mapping.values() if v['role']=='decode')
        end=min(right,left+.2);intervals=[]
        for row in csv.DictReader(trace.open()):
            if int(row['device'])!=device:continue
            a,b=host(int(row['start_ns'])),host(int(row['end_ns']))
            if b>left and a<end:intervals.append((max(left,a),min(end,b)))
        axes[index,1].broken_barh([(1000*(a-left),1000*(b-a)) for a,b in union(intervals)],(0,1),facecolors='#31567a')
        axes[index,1].set(xlim=(0,1000*(end-left)),ylim=(-.2,1.6),xlabel='First fixed 200 ms (ms)',yticks=[],ylabel='Observed graph envelopes\n+ eager kernels')
        axes[index,1].text(.02,.86,'Safe future slack: NOT_MEASURED',transform=axes[index,1].transAxes,fontsize=9)
        quality='QUALIFIED' if record['qualified_instrumentation'] else 'DIAGNOSTIC: interference gate failed'
        axes[index,1].set_title(model.capitalize()+' — '+quality,fontsize=9)
        for filename,label in (('compute_free_lower_bound_gaps.csv','No-compute lower bound'),
                ('compute_free_upper_envelope_gaps.csv','Optimistic upper envelope')):
            path=folder/filename;sources.append(path)
            values=sorted(float(v['duration_ms']) for v in csvrows(path) if v['same_run_P_backlog']=='True')
            if values:axes[index,2].plot(values,np.arange(1,len(values)+1)/len(values),label=label)
        first=costs[model][0]
        minimum=min(first['streams']['decode']['per_request_sum_ms'].values())
        axes[index,2].axvline(minimum,color='#af3333',ls='--',label='Best observed full-prompt 1-loop cost')
        axes[index,2].set(xscale='log',xlabel='Gap or cost (ms)',ylabel='CDF',ylim=(0,1.03));axes[index,2].legend(fontsize=7)
    fig.suptitle('Native activity and same-run P pressure — bounds are not Safe Slack',fontsize=14)
    save(fig,'figure_1_native_activity_idle_bounds',sources,
        'Measured graph/eager activity and host pressure. Graph complements bound physical idle. Future deadline-safe slack remains unmeasured. Diagnostic traces remain labeled.')

    # 2: Oracle diagnostics must not masquerade as actual adaptive decisions.
    fig,axes=plt.subplots(1,2,figsize=(12,4));path=ROOT/'offline_budget_loop_decisions.csv';rows=csvrows(path)
    for ax,model in zip(axes,('ouro','huginn')):
        data=[r for r in rows if r['model']==model]
        if data:
            stride=max(1,len(data)//1000);data=data[::stride]
            ax.scatter([float(r['optimistic_budget_ms']) for r in data],[int(r['selected_loops']) for r in data],s=8,alpha=.3)
            ax.set(xlabel='Offline optimistic gap budget (ms)',ylabel='Cost-fit remaining loops',xscale='log')
        else:missing(ax,'No validated online budget or cut decisions')
        ax.set_title(model.capitalize()+' — OFFLINE ONLY')
    fig.suptitle('Online variable Loop selection: NOT_MEASURED',color='#7a263a')
    save(fig,'figure_2_offline_budget_loop_fit',[path],
        'Clairvoyant relaxed cost-fit diagnostics only. No point represents an actual accepted adaptive serving handoff.')

    # 3: A real serial request, not a constructed dynamic serving timeline.
    fig,axes=plt.subplots(2,2,figsize=(13,6));sources=[]
    for i,model in enumerate(('ouro','huginn')):
        for j,loops in enumerate((1,2)):
            row=next(v for v in costs[model] if v['remaining_loops']==loops)
            folder=ROOT/'raw'/row['name'];file=folder/'service_latency_requests.json';sources.append(file)
            request=load(file)[0];origin=request['arrival'];p=request['actual_P_model_stream_entry']['time']
            d=request['first_token']-request['D_continuation_to_first_token_ms']/1000
            ax=axes[i,j]
            ax.broken_barh([(1000*(p-origin),request['P_prompt_device_enclosure_ms'])],(2,.6),facecolors='#31567a')
            ax.broken_barh([(1000*(d-origin),request['D_prompt_device_enclosure_ms'])],(0,.6),facecolors='#c57925')
            closed=folder/'closed_stats.json';sources.append(closed)
            timing=next(r for r in load(closed)['metrics'] if r['request_id']==request['request_id'])
            fields=timing['timings'];created=timing['created_monotonic']
            if 'prefill_compute_completed' in fields and 'd_received' in fields:
                a,b=created+fields['prefill_compute_completed'],created+fields['d_received']
                ax.broken_barh([(1000*(min(a,b)-origin),1000*abs(b-a))],(1,.6),facecolors='#9b784f',
                    hatch='xx' if b<a else None)
            ax.axvline(1000*(request['first_token']-origin),color='#2e7d32',ls='--',label='First output delivery')
            ax.set(yticks=(.3,1.3,2.3),yticklabels=('D prompt enclosure','Host handoff receipts','P prompt enclosure'),xlabel='Since request arrival (ms)',
                title=f'{model.capitalize()}: {loops} remaining loops — serial forced cut')
            ax.legend(fontsize=8)
    fig.suptitle('Actual CUDA-event enclosures — not adaptive, not concurrent Decode')
    save(fig,'figure_3_actual_serial_continuation',sources,
        'Actual first measured serial requests. Event enclosures include waits. Native transfer precedes D entry. No fabricated per-kernel or adaptive schedule.')

    # 4: Keep unexecuted arms explicit when an earlier gate blocks expansion.
    runs=csvrows(ROOT/'serving_runs.csv');slos=load(ROOT/'configs/frozen_SLO.json')
    fig,axes=plt.subplots(2,4,figsize=(16,7));sources=[ROOT/'serving_runs.csv']
    for i,model in enumerate(('ouro','huginn')):
        names=[f'a_{model}_{load}_off_load_control_n128' for load in ('low','medium')]+[f'a_{model}_heavy_off_s1_n128']
        data=[next((r for r in runs if r['name']==name),None) for name in names]
        data=[r for r in data if r]
        for ax,metric,label in zip(axes[i],('TTFT_p95_ms','ITL_p99_ms','E2E_p95_ms','joint_goodput'),
                ('TTFT p95 (ms)','ITL p99 (ms)','E2E p95 (ms)','Finite joint SLO-goodput (req/s)')):
            if data:ax.plot([float(r['nominal_QPS']) for r in data],[float(r[metric]) for r in data],'-o',label='Native Atomic PD')
            else:missing(ax)
            ax.set(xlabel='Offered QPS',ylabel=label,title=model.capitalize())
            ax.text(.03,.94,'A1–A4: NOT_RUN / gate-dependent',transform=ax.transAxes,va='top',fontsize=8,color='#7a263a')
        axes[i,0].axhline(slos[f'{model}_L{4096 if model=="ouro" else 2048}']['TTFT_ms'],color='#af3333',ls='--')
        axes[i,1].axhline(slos[f'{model}_L{4096 if model=="ouro" else 2048}']['ITL_ms'],color='#af3333',ls='--')
    fig.suptitle('Current Native load controls — no unmeasured method speedups')
    save(fig,'figure_4_native_load_controls_and_missing_arms',sources,
        'Measured finite Native cohorts. No Dynamic Loop/Token performance is invented. Primary marker-free Service TTFT remains unmeasured.')

    # 5: Equal recurrent rows do not imply equal attention work or equal FLOPs.
    extra=load(ROOT/'equal_quantum_primitive_costs.json') if (ROOT/'equal_quantum_primitive_costs.json').exists() else []
    fig,axes=plt.subplots(2,4,figsize=(16,6));sources=[ROOT/'equal_quantum_primitive_costs.json',ROOT/'equal_quantum_operation_enclosures.csv']
    op_rows=csvrows(ROOT/'equal_quantum_operation_enclosures.csv')
    for i,model in enumerate(('ouro','huginn')):
        data=[r for r in extra if r['model']==model and r['passed']]
        data.sort(key=lambda r:bool(r.get('frontier')))
        labels=['Token suffix' if r.get('frontier') else 'Whole-prompt Loop' for r in data]
        metrics=[([r['streams']['decode']['p50_ms'] for r in data],'D total prompt enclosure (ms)'),
            ([max(float(op['stream_enclosure_ms']) for op in op_rows if op['name']==r['name'] and op['role']=='decode')
                for r in data],'Largest D submission enclosure (ms)'),
            ([r['counters']['prefill']['transport_bytes_sent']/r['count']/1024**3 for r in data],'Actual state payload (GiB)'),
            ([r['service_TTFT']['p50_ms'] for r in data],'Serial Service TTFT p50 (ms)')]
        for ax,(values,label) in zip(axes[i],metrics):
            if values:ax.bar(labels,values,color=['#c57925','#31567a'][:len(values)])
            else:missing(ax,'Equal-quantum cost not yet measured')
            ax.set(ylabel=label,title=model.capitalize());ax.tick_params(axis='x',labelrotation=12)
    fig.suptitle('Equal D loop quantum — serial diagnostics, not serving gains')
    save(fig,'figure_5_equal_quantum_token_loop_cost',sources,
        'Measured state-qualified serial costs when available. Same recurrent rows are not equal FLOPs. All costs include the existing numerical fixes.')

    # 6: Separate enclosures. Do not add these medians into a critical path.
    fig,axes=plt.subplots(2,3,figsize=(14,6));sources=[ROOT/'full_prompt_primitive_costs.json']
    for i,model in enumerate(('ouro','huginn')):
        native=next(r for r in primitives if r['model']==model and r['remaining_loops']==0)
        data=costs[model];loop_counts=[r['remaining_loops'] for r in data]
        p0=native['streams']['prefill']['per_request_sum_ms']
        saving=[pct([p0[str(seed)]-value for seed,value in r['streams']['prefill']['per_request_sum_ms'].items()],50) for r in data]
        for ax,values,label in zip(axes[i],(saving,[r['streams']['decode']['p50_ms'] for r in data],[r['handoff']['p50_ms'] for r in data]),
                ('P enclosure reduction p50 (ms)','D additional prompt enclosure p50 (ms)','Observed host handoff p50 (ms)')):
            ax.plot(loop_counts,values,'-o');ax.set(xlabel='Remaining loops on D',ylabel=label,title=model.capitalize())
    fig.suptitle('Independent measured components — not an additive speedup model')
    save(fig,'figure_6_serial_cost_components',sources,
        'P reductions use matched-request serial enclosure differences. D and handoff show separate distributions. Do not add component percentiles or infer online TTFT gains.')

    # 7: Cost-fit, not SLO-safe, and no invented tiled area.
    fig,axes=plt.subplots(1,2,figsize=(13,4));budgets=[1,2,4,8,16,32,64,128,256]
    for ax,model in zip(axes,('ouro','huginn')):
        data=costs[model]
        minimum=[min(r['streams']['decode']['per_request_sum_ms'].values()) for r in data]
        matrix=np.array([[float(b>=c) for b in budgets] for c in minimum])
        ax.imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap='Blues',origin='lower')
        ax.set(xticks=range(len(budgets)),xticklabels=budgets,yticks=range(len(data)),
            yticklabels=[r['remaining_loops'] for r in data],xlabel='Hypothetical available budget (ms)',ylabel='Full-prompt remaining loops',title=model.capitalize())
        for y,c in enumerate(minimum):ax.text(len(budgets)-1,y,f'{c:.1f} ms',ha='right',va='center',fontsize=8)
    fig.suptitle('Compute-only cost-fit — SLO-safe region and tiled continuation: NOT_MEASURED',color='#7a263a')
    save(fig,'figure_7_cost_fit_not_safe_region',[ROOT/'full_prompt_primitive_costs.json'],
        'Measured best-observed serial costs compared with hypothetical budgets. Handoff and safety are absent. Blue cells do not establish feasible or SLO-safe offloading.')
    write(ROOT/'figure_manifest.json',dict(time=__import__('time').time(),figures=manifest))
    print(json.dumps(dict(figures=len(manifest))))


if __name__=='__main__':
    initialize()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('idle_campaign_figures'):main()
