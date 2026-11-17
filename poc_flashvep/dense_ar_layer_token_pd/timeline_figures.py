"""Keep host boundaries separate from CUDA stream enclosures."""
import json
from .common import ROOT,write
from .analysis import records


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=[]
    screens={r['method']:r for r in json.loads((ROOT/'analysis/CUDA_event_interference_screen.json').read_text())}
    for method in ('Static','Token'):
        name=f'enclosure_{method}_events_N144_v1';folder=ROOT/'raw'/name
        window=json.loads((folder/'window.json').read_text());origin=window['start']
        selected={'q'+str(i) for i in range(16)}
        fig,axes=plt.subplots(2,1,figsize=(11,5),sharex=True,
            gridspec_kw={'height_ratios':[1,1.3]})
        samples=[];anchors={}
        for role,y in (('prefill',1),('decode',0)):
            obj=json.loads((folder/f'CUDA_event_enclosures_{role}.json').read_text())
            anchors[role]=obj['anchor_uncertainty_ms']
            for row in obj['rows']:
                ids=selected.intersection(row['requests'])
                if not ids:
                    continue
                start=row['start_monotonic']-origin;end=row['end_monotonic']-origin
                kind='P prefix' if role=='prefill' else 'D suffix' if row['prefilling'] else 'D decode'
                color={'P prefix':'#377eb8','D suffix':'#984ea3','D decode':'#e41a1c'}[kind]
                axes[0].broken_barh([(start,end-start)],(y-.25,.5),facecolors=color)
                if row['prefilling']:
                    axes[0].text(start,y+.28,','.join(sorted(ids,key=lambda s:int(s[1:]))),fontsize=6)
                samples.append(dict(kind=kind,requests=row['requests'],start_s=start,end_s=end,
                    event_elapsed_ms=row['elapsed_ms'],graph_replay=row.get('graph_replay'),
                    query_tokens=row['query_tokens'],stream='NOT_MEASURED'))
        axes[0].set_yticks([0,1],['D stream enclosure','P stream enclosure'])
        axes[0].set_ylim(-.6,1.7)
        axes[0].set_title('CUDA-event enclosures: include stream gaps; not exclusive kernels\n'+
            f"Host anchor bracket: P {anchors['prefill']:.4f} ms, D {anchors['decode']:.4f} ms",fontsize=9)
        submissions={r['request_id']:r['actual_send_start'] for r in records(folder/'actual_submissions.jsonl')}
        dsub={r['request_id']:r['actual_send_start'] for r in records(folder/'D_actual_submissions.jsonl')}
        ready={r['request_id']:r['time'] for r in records(folder/'native_handoffs.jsonl')}
        requests={'q'+str(r['sample_id']):r for r in records(folder/'requests.jsonl')}
        host=[]
        for i in range(16):
            rid='q'+str(i);row=requests[rid]
            boundaries=(('P submit',submissions[rid]),('Source ready',ready[rid]),
                ('D submit',dsub[rid]),('First output',row['first_token']))
            axes[1].plot([ready[rid]-origin,dsub[rid]-origin],[i,i],color='#aaaaaa',lw=3)
            for kind,timestamp in boundaries:
                marker,color={'P submit':('>','#377eb8'),'Source ready':('|','#377eb8'),
                    'D submit':('>','#984ea3'),'First output':('*','#111111')}[kind]
                axes[1].plot(timestamp-origin,i,marker=marker,color=color,ms=5,
                    label=kind if i==0 else None,linestyle='none')
            host.append(dict(request=rid,boundaries={kind:t-origin for kind,t in boundaries},
                source_ready_to_D_submit_ms=1000*(dsub[rid]-ready[rid])))
        axes[1].set_yticks(range(16),['q'+str(i) for i in range(16)],fontsize=7)
        axes[1].invert_yaxis();axes[1].legend(fontsize=7,ncol=4)
        axes[1].set(xlabel='Seconds from this run\'s first offered arrival',
            title='Host boundaries; gray span = ready-to-submit queue + handoff, not wire time')
        axes[1].title.set_fontsize(9)
        passed=screens[method]['passed']
        fig.suptitle(f'{method}, matched Last12 work; capture interference screen '+('PASS' if passed else 'FAIL — diagnostic only'),fontsize=11)
        fig.tight_layout()
        stem=f'11_{method.lower()}_host_stream_timeline'
        for suffix in ('.png','.pdf'):
            fig.savefig(ROOT/'figures'/(stem+suffix),dpi=180,bbox_inches='tight')
        plt.close(fig)
        results.append(dict(name=name,stem=stem,origin_monotonic=origin,
            anchor_uncertainty_ms=anchors,screen=screens[method],CUDA_enclosures=samples,host=host,
            evidence='TRACE_ANALYSIS',physical_kernel_intervals='NOT_MEASURED',
            scope='Each figure uses one actual server start. Separate runs are not merged. No stream-ID, SM/HBM counter or physical idle claim. Event edges estimate stream enclosure boundaries, not individual kernel start/end.'))
    write(ROOT/'analysis/host_stream_timeline_data.json',results)
    print('HOST_STREAM_TIMELINES',len(results),flush=True)


if __name__=='__main__':
    main()
