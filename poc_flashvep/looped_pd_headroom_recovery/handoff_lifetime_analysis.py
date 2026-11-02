"""Report observed handoff and source-retention spans without calling them DMA."""
import collections
import statistics
from .common import *
from .analyze import records,table
from .reporting import FINAL,md,fmt


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.size':9})
    plan=json.loads((ROOT/'state/confirmation_plan.json').read_text())
    all_rows=[];summary=[];figures=[]
    phase_keys=[('admission_wait',None,'admitted'),('D_reservation','admitted','d_reserved'),
        ('P_start_wait','d_reserved','prefill_started'),('P_compute_observation','prefill_started','prefill_compute_completed'),
        ('compute_end_to_commit','prefill_compute_completed','prefill_submitted'),
        ('commit_to_D_received','prefill_submitted','d_received'),
        ('D_activation_wait','d_received','d_activated'),('D_first_token_wait','d_activated','first_token'),
        ('generation_after_first','first_token','finished')]
    for item in plan['runs']:
        name=item.get('PD')
        folder=ROOT/'raw'/name
        status=folder/'dispatcher_result.json'
        if not status.exists() or not json.loads(status.read_text()).get('complete'):continue
        metrics=json.loads((folder/'closed_stats.json').read_text())['metrics']
        lifetime=records(folder/'lifetime_events.jsonl')
        released={r['request_id']:r['time'] for r in lifetime if r['event']=='COORDINATOR_released' and r['role']=='prefill'}
        measured=[r for r in metrics if r['request_id'].startswith('r0_')]
        for r in measured:
            t=r['timings'];base=r['created_monotonic']
            for phase,start,end in phase_keys:
                if end not in t or start is not None and start not in t:continue
                begin=0 if start is None else t[start]
                all_rows.append(dict(name=name,model=item['model'],independent_start=item['start'],
                    request_id=r['request_id'],phase=phase,begin_monotonic=base+begin,end_monotonic=base+t[end],
                    host_span_ms=1000*(t[end]-begin),share_of_request_E2E_percent=100*(t[end]-begin)/t['finished'],
                    nonnegative_span=t[end]>=begin,evidence='MEASURED_COORDINATOR_MESSAGE_RECEIPT_INTERVAL',
                    scope='Host message boundaries. GPU work, communication, queueing, and other requests can overlap these spans. These are not exclusive device-operation percentages.'))
            if r['request_id'] in released and 'prefill_compute_completed' in t:
                all_rows.append(dict(name=name,model=item['model'],independent_start=item['start'],request_id=r['request_id'],
                    phase='P_compute_end_to_source_release_nonadditive',begin_monotonic=base+t['prefill_compute_completed'],
                    end_monotonic=released[r['request_id']],host_span_ms=1000*(released[r['request_id']]-base-t['prefill_compute_completed']),
                    share_of_request_E2E_percent=None,nonnegative_span=released[r['request_id']]>=base+t['prefill_compute_completed'],
                    evidence='MEASURED_NATIVE_SOURCE_RELEASE_AFTER_ACK',
                    scope='Source ownership remains until native ACK handling. This overlaps other phases and must not be added to E2E shares.'))
        if item['start']==1 and measured:
            chosen=sorted(measured,key=lambda r:r['created_monotonic'])[48:64]
            if not chosen:chosen=measured[:16]
            origin=min(r['created_monotonic'] for r in chosen)
            fig,ax=plt.subplots(figsize=(12,6));colors=plt.get_cmap('tab10')
            labels=set()
            for i,r in enumerate(chosen):
                t=r['timings'];base=r['created_monotonic']
                ax.plot(base-origin,i,marker='v',color='black',ms=4,label='Arrival' if 'Arrival' not in labels else None);labels.add('Arrival')
                for j,(phase,start,end) in enumerate(phase_keys):
                    if end not in t or start is not None and start not in t:continue
                    begin=0 if start is None else t[start]
                    if t[end]<begin:continue
                    ax.broken_barh([(base+begin-origin,t[end]-begin)],(i-.35,.7),facecolors=colors(j),
                        label=phase if phase not in labels else None)
                    labels.add(phase)
            ax.set_yticks(range(len(chosen)),[r['request_id'] for r in chosen])
            ax.set(xlabel='Seconds after first displayed arrival',ylabel='Measured request',
                title=f'{item["model"].title()} | N128 start 1, requests 49–64 | HOST phase observations, not GPU execution')
            ax.grid(axis='x',alpha=.2);ax.legend(fontsize=7,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.14));fig.tight_layout()
            outputs=[]
            for suffix in ('png','svg'):
                p=ROOT/'figures'/(item['model']+'_native_request_handoff_timeline.'+suffix)
                fig.savefig(p,dpi=180,bbox_inches='tight');outputs.append(str(p.relative_to(ROOT)))
            plt.close(fig)
            sources=[folder/'closed_stats.json',folder/'lifetime_events.jsonl']
            figures.append(dict(name=item['model']+'_native_request_handoff_timeline',outputs=outputs,
                exact_sources=[str(p.relative_to(ROOT)) for p in sources],source_hashes={str(p.relative_to(ROOT)):digest(p) for p in sources},
                scope='Actual host event intervals only. No blank interval is labeled GPU idle. Source release requires native ACK.'))
    grouped=collections.defaultdict(list)
    for r in all_rows:grouped[(r['name'],r['model'],r['phase'])].append(r)
    for (name,model,phase),values in grouped.items():
        durations=sorted(r['host_span_ms'] for r in values);i=.95*(len(durations)-1);a=int(i);b=min(a+1,len(durations)-1)
        summary.append(dict(name=name,model=model,phase=phase,requests=len(values),
            mean_host_span_ms=statistics.mean(durations),p95_host_span_ms=durations[a]+(i-a)*(durations[b]-durations[a]),
            nonnegative_spans=all(r['nonnegative_span'] for r in values),evidence='MEASURED_HOST_PHASE_SUMMARY',
            scope='One process start. No independent-start confidence interval is inferred from the request count.'))
    table('native_handoff_host_spans.csv',all_rows);table('native_handoff_host_summary.csv',summary)
    manifest=json.loads((ROOT/'figure_manifest.json').read_text());manifest['figures'].extend(figures);write(ROOT/'figure_manifest.json',manifest)
    display=[[r['model'],r['name'],r['phase'],r['requests'],fmt(r['mean_host_span_ms']),fmt(r['p95_host_span_ms'])]
        for r in summary if r['phase'] in ('P_compute_observation','compute_end_to_commit','commit_to_D_received','P_compute_end_to_source_release_nonadditive')]
    text='\n## Native handoff and source-retention observations\n\n'
    text+=md(['Model','Independent run','Host interval','N','Mean ms','P95 ms'],display)+'\n\n'
    text+='These intervals use coordinator message receipts. They are not exclusive DMA or GPU compute measurements.\n'
    text+='The compute-end-to-commit span can include descriptor preparation, submission, transport progress, and message delivery.\n'
    text+='Source-release spans retain native ACK dependencies. They overlap the additive host phases.\n'
    text+='A long source-retention span can consume a P staging slot after computation ends.\n'
    text+='The data do not prove that removing all such spans is safe or achieves an equivalent latency reduction.\n'
    text+='No early release or unacknowledged state reclamation is implemented.\n'
    text+='The request timeline displays host observations. The separate kernel timeline displays actual observed device activity.\n'
    FINAL.write_text(FINAL.read_text()+text)
    print(json.dumps(dict(request_phase_rows=len(all_rows),summary_rows=len(summary),figures=len(figures))))


if __name__=='__main__':main()
