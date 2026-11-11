"""Plot actual observations and mark gated, unrun comparisons explicitly."""
import csv,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

H='profile_huginn_L3072_O16_P2_mu115_N64_a2026111230_pinned_capture'


def read(path):return json.loads(path.read_text())


def intervals(path):
    with path.open() as f:return [(float(r['start']),float(r['end'])) for r in csv.DictReader(f)]


def main():
    initialize();assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    out=ROOT/'figures';out.mkdir(exist_ok=True);manifest=[]
    def save(fig,name,scope,sources):
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=170);fig.savefig(out/(name+'.svg'));plt.close(fig)
        manifest.append(dict(name=name,scope=scope,sources=sources,files=[name+'.png',name+'.svg']))
    folder=ROOT/'raw'/H;trace=read(folder/'physical_interval_analysis.json');left=trace['decode']['left'];right=trace['decode']['right']
    kernel={r:intervals(folder/f'physical_{r}_union.csv') for r in ('prefill','decode')}
    idle=intervals(folder/'physical_decode_idle_windows.csv')
    queue=read(folder/'submission_queue_audit_timeline.json')
    source=read(folder/'discovery_events_prefill.json')['rows']
    cuts=[r for r in source if r.get('depth')==31 and left<=r.get('gpu_end_host_estimate',-1)<=right]
    fig,ax=plt.subplots(3,1,figsize=(12,5.8))
    for a,role,color in zip(ax[:2],('prefill','decode'),('#dc962f','#3179b4')):
        # Rasterization preserves actual interval positions without millions of SVG paths.
        a.broken_barh([(max(s,left)-left,min(e,left+.2)-max(s,left)) for s,e in kernel[role] if s<left+.2 and e>left],(0,1),facecolors=color,rasterized=True)
        a.set_yticks([.5],['P kernels' if role=='prefill' else 'D kernels']);a.set_ylim(-.2,1.2)
        a.set_xlim(0,.2);a.set_xlabel('Seconds from common capture start (first 200 ms)')
    t=[r['time']-left for r in queue];q=[r['audited_count'] for r in queue]
    ax[2].step(t,q,where='post',color='#555555');ax[2].set_ylabel('Audited P\nready count');ax[2].set_xlabel('Seconds from common capture start')
    ax[0].set_title('Huginn 3072/16 Native: one qualified 7.408 s pilot\nActual all-stream kernel union. Not a repeated Phase 1 qualification.')
    ax[2].set_xlim(0,right-left)
    save(fig,'figure_1_huginn_actual_kernel_timeline','MEASURED / TRACE_ANALYSIS: one qualified profiling pilot, not Decode-safe capacity',[H+'/physical_interval_analysis.json',H+'/submission_queue_audit_timeline.json'])
    edges=np.linspace(left,right,31);busy=[]
    for l,r in zip(edges,edges[1:]):busy.append(100*sum(max(0,min(b,r)-max(a,l)) for a,b in kernel['prefill'])/(r-l))
    fig,ax=plt.subplots(2,1,figsize=(10,5),sharex=True)
    ax[0].stairs(busy,edges-left,color='#dc962f');ax[0].axhline(80,color='k',ls='--',lw=.8);ax[0].set_ylabel('P kernel union %');ax[0].set_ylim(0,105)
    ax[1].step(t,q,where='post');ax[1].set_ylabel('Audited P ready');ax[1].set_xlabel('Seconds from common capture start')
    ax[0].set_title('Same run and common window. Host queue and physical busy remain distinct.')
    save(fig,'figure_2_same_window_P_pressure','MEASURED / TRACE_ANALYSIS: kernel unions per fixed bin and audited host ownership',[H+'/physical_prefill_union.csv',H+'/submission_queue_audit_timeline.json'])
    durations=np.array([(b-a)*1000 for a,b in idle]);thresholds=[1,2,5,10,20,30,50,100]
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    ax[0].hist(durations,bins=np.logspace(-6,1,60));ax[0].set_xscale('log');ax[0].set_yscale('log');ax[0].set_xlabel('Actual physical kernel gap (ms)');ax[0].set_ylabel('Count')
    ax[1].bar([str(v) for v in thresholds],[sum(durations>=v) for v in thresholds]);ax[1].set_xlabel('Threshold (ms, overlapping sets)');ax[1].set_ylabel('Gaps at least threshold')
    ax[0].set_title('748231 gaps; maximum 2.427 ms')
    ax[1].set_title('Inactivity is not an admission window')
    save(fig,'figure_3_huginn_idle_distribution','MEASURED: one qualified pilot. Ouro qualified idle distribution is NOT_MEASURED',[H+'/physical_decode_idle_windows.csv'])
    largest=sorted(idle,key=lambda r:r[1]-r[0],reverse=True)[:110]
    fig,ax=plt.subplots(2,1,figsize=(11,4.4),sharex=True)
    ax[0].broken_barh([(a-left,b-a) for a,b in largest],(0,1),facecolors='#438c76');ax[0].set_yticks([.5],['D gaps >=1 ms'])
    ax[1].eventplot([r['gpu_end_host_estimate']-left for r in cuts],lineoffsets=.5,linelengths=.7,color='#dc962f');ax[1].set_yticks([.5],['P R31 ready event'])
    ax[1].set_xlabel('Seconds from common capture start');ax[1].set_xlim(0,right-left)
    ax[0].set_title('Measured gap positions and mapped P events\nNo exported source snapshot or counterfactual handoff. Not verified feasible migration.')
    save(fig,'figure_4_source_readiness_and_gaps','TRACE_ANALYSIS: source event clock drift is not bounded. No feasible cut is inferred',[H+'/discovery_events_prefill.json',H+'/physical_decode_idle_windows.csv'])
    inv=read(ROOT/'analysis/final_campaign_inventory.json')
    starts=[r for r in inv['serving'] if r['configuration']['name'].startswith('confirm_v2_huginn_L3072_O16_P2_native_N128_')]
    if not starts:
        starts=[r for r in inv['serving'] if r['configuration']['name'].startswith('confirm_huginn_L3072_O16_P2_native_N128_')]
    # Select only the three original marker-free N128 controls identified by the immutable plan.
    plan=read(ROOT/'state/huginn_native128_plan.json');names=set(plan['jobs'])
    starts=[r for r in inv['serving'] if r['configuration']['name'] in names and not r['configuration']['instrumented']]
    fig,ax=plt.subplots(1,3,figsize=(12,4))
    for a,key,title in zip(ax,('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms'),('TTFT p95','ITL p99','TPOT p95')):
        a.plot(range(1,len(starts)+1),[r['metrics'][key] for r in starts],'o-');a.set_title(title+' (ms)');a.set_xlabel('Independent Native start');a.set_xticks(range(1,len(starts)+1))
        a.text(.04,.96,'Conservative migration: NOT_RUN',va='top',transform=a.transAxes,fontsize=8)
    fig.suptitle('Huginn Native confirmation. No Candidate/Native speedup was measured.')
    save(fig,'figure_5_native_confirmation_and_gated_candidate','MEASURED Native only. Migration comparison NOT_RUN',[str((ROOT/'state/huginn_native128_plan.json').relative_to(ROOT))])
    packing=read(folder/'fixed_trace_enclosure_packing.json')
    rows=[r for r in packing['cells'] if r['cost_model']=='MODEL_RUNNER_QUANTUM_ENCLOSURE' and r['cost_quantile']=='p50']
    fig,ax=plt.subplots(1,2,figsize=(11,4.5))
    labels=[str(r['token_tile']) for r in rows]
    ax[0].bar(labels,[r['cost_ms'] for r in rows],color='#dc962f');ax[0].axhline(max(durations),color='#3179b4',ls='--',label='Largest measured D gap');ax[0].set_ylabel('Milliseconds');ax[0].set_xlabel('Tokens per one-loop quantum');ax[0].legend(fontsize=8)
    ax[1].axis('off');ax[1].text(.02,.95,'OFFLINE_ORACLE: fixed Native trace\n\nAll four measured Runner cost models:\n0 completed full loops\n\nCore-only tile128 optimism:\n1 tile slot / 24 tiles per loop\n0 completed full loops\n\nMigration-fraction / TTFT–ITL frontier:\nNOT_RUN',va='top',fontsize=11)
    ax[0].set_title('Serial CUDA Event enclosure, not kernel-only time')
    save(fig,'figure_6_fixed_trace_budget_screen','OFFLINE_ORACLE: zero handoff and free state optimism. Not a universal physical bound or online performance',[H+'/fixed_trace_enclosure_packing.json'])
    write(out/'figure_manifest.json',dict(figures=manifest,
        missing=dict(ouro_qualified_kernel_timeline='NOT_MEASURED',
            conservative_migration_comparison='NOT_RUN',migration_tradeoff='NOT_RUN'),
        scope='No conceptual or invented GPU intervals appear. Raw data, diagnostic traces, and Oracle calculations keep separate labels.'))
    print(json.dumps(dict(figures=len(manifest))))


if __name__=='__main__':main()
