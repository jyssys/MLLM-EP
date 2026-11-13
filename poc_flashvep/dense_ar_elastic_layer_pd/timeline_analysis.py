"""Plot observed stream enclosures without inferring physical inactivity."""
import io
from .common import *
from .analyze import records


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(4,1,figsize=(17,10),sharex=True)
    evidence=[]
    candidates=records(ROOT/'raw/enclosure_Revised_O16_N64_events_v2/requests.jsonl')
    target=next(r['sample_id'] for r in sorted(candidates,key=lambda r:r['sample_id'])
        if r.get('completed_D_prefill_layers',0)>0 and r['sample_id']>=len(candidates)//2)
    for ax,arm in zip(axes,('Native','Colocated','Static','Revised')):
        name=f'enclosure_{arm}_O16_N64_events_v2';raw=ROOT/'raw'/name
        window=json.loads((raw/'window.json').read_text());origin=window['start']
        requests=records(raw/'requests.jsonl');rows=[]
        for path in raw.glob('CUDA_event_enclosures_*.json'):
            role=path.stem.removeprefix('CUDA_event_enclosures_')
            if role.startswith('replica'):lane=0 if role.endswith('0') else 1
            else:lane=0 if role=='prefill' else 1
            value=json.loads(path.read_text())
            for event in value['rows']:
                ids=event.get('requests',[])
                if not ids or not all(q.startswith('q') and q[1:].isdigit() for q in ids):continue
                start=event['start_monotonic']-origin;end=event['end_monotonic']-origin
                suffix=(role=='decode' and event.get('layer_mode')=='MIGRATE' and event.get('prefilling')
                    and event.get('cut',36)<36 and event['query_tokens']>len(ids))
                kind='P prefix' if role=='prefill' else 'D suffix' if suffix else 'Decode' if event['query_tokens']<=len(ids) else 'Colocated prefill/mixed'
                color={'P prefix':'#dda147','D suffix':'#128e83','Decode':'#397db5','Colocated prefill/mixed':'#a66c94'}[kind]
                ax.broken_barh([(start,end-start)],(lane-.18,.36),facecolors=color,edgecolors='none',alpha=.8)
                rows.append(dict(kind=kind,lane=lane,start=start,end=end,requests=ids,cut=event.get('cut'),
                    query_tokens=event['query_tokens']))
        sample=next(r for r in requests if r['sample_id']==target)
        rid='q'+str(sample['sample_id'])
        ax.axvline(sample['arrival']-origin,color='#777777',ls=':',lw=.8)
        ax.axvline(sample['first_token']-origin,color='#bd3442',lw=1.1)
        ax.text(sample['first_token']-origin,.62,f"{rid} first output: {sample['ttft_ms']/1000:.2f}s TTFT",fontsize=8)
        deadlines=[]
        for previous,current in zip(sample['token_events'],sample['token_events'][1:]):
            deadline=previous['time']+.034
            ax.plot([deadline-origin],[1.22],marker='|',markersize=4,color='#bd3442' if current['time']>deadline else '#555555')
            deadlines.append(dict(time=deadline-origin,delivery_miss=current['time']>deadline))
        handoffs={r['request_id']:r['time'] for r in records(raw/'native_handoffs.jsonl')}
        observed=[e['start'] for e in rows if e['lane']==1 and rid in e['requests']]
        if rid in handoffs and observed:
            lo=handoffs[rid]-origin;hi=min(observed)
            if hi>=lo:ax.broken_barh([(lo,hi-lo)],(.38,.11),facecolors='#888888',hatch='//')
        frontier=[];boundary_path=raw/'source_layer_boundary_events.json'
        if boundary_path.exists():
            actual_cut=sample.get('actual_layer_cut',36)
            for event in json.loads(boundary_path.read_text())['rows']:
                if rid in event['requests'] and event['completed_layer']==actual_cut:
                    x=event['time']-origin;ax.plot(x,.25,marker='^',color='#b57a24',markersize=5)
                    frontier.append(dict(time=x,completed_layer=actual_cut,scope='CUDA stream frontier, not kernel end'))
        transfer=[]
        for path in raw.glob('native_decode_*.json'):
            for event in json.loads(path.read_text()).get('NIXL_done_events',[]):
                if any(q.startswith(rid+'-') for q in event['request_ids']):
                    x=event['time']-origin;ax.plot(x,.55,marker='D',color='#555555',markersize=4)
                    transfer.append(dict(time=x,scope='Host-observed Native KV completion'))
        acknowledgements=[]
        for path in (raw/'state_transport').glob(rid+'-*.hidden_ack.json'):
            event=json.loads(path.read_text());x=event['copy_complete']-origin
            ax.plot(x,.75,marker='s',color='#9c59a0',markersize=4)
            acknowledgements.append(dict(time=x,scope='Host Hidden-copy completion and ACK'))
        ax.set_yticks([0,1],['GPU0 replica','GPU1 replica'] if arm=='Colocated' else ['GPU0 / P','GPU1 / D']);ax.set_ylim(-.4,1.45)
        ax.set_title(arm+' — DIAGNOSTIC stream-event enclosures, not exclusive kernel intervals',fontsize=10)
        ax.grid(axis='x',alpha=.2)
        evidence.append(dict(name=name,origin=origin,selected_request=rid,events=rows,
            request_first_output=sample['first_token']-origin,TTFT_ms=sample['ttft_ms'],host_delivery_deadlines=deadlines,
            source_frontier=frontier,KV_completion=transfer,Hidden_ACK=acknowledgements))
    axes[-1].set_xlabel('Seconds from each run measurement start (same prompt/arrival trace, separate starts)')
    fig.legend(handles=[Patch(color=c,label=k) for k,c in [('P prefix','#dda147'),('D suffix','#128e83'),
        ('Decode','#397db5'),('Colocated prefill/mixed','#a66c94'),('Host ready-to-D-execution wait, including handoff','#888888')]],
        loc='lower center',ncol=3,fontsize=8)
    fig.suptitle('Actual observed execution — blank space is NOT measured GPU idle\nUnqualified diagnostic. Upper ticks: Host deadlines. Triangle: CUDA source frontier. Diamond: Host KV done. Square: Host Hidden ACK.',fontsize=11)
    fig.tight_layout(rect=(0,.07,1,.94))
    fig.savefig(folder/'figure_I_diagnostic_PD_timeline.png',dpi=180,bbox_inches='tight')
    fig.savefig(folder/'figure_I_diagnostic_PD_timeline.svg',bbox_inches='tight')
    output=io.StringIO();fig.savefig(output,format='svg',bbox_inches='tight');plt.close(fig)
    html='''<!doctype html><meta charset="utf-8"><title>Observed PD timeline — diagnostic</title>
<h1>Observed PD execution</h1><p>Diagnostic CUDA stream enclosures and Host observations only.
Blank space is not measured physical idle. The 5% tail-interference gate failed.
Each panel uses its own measurement origin. No synthetic TO-BE events appear.</p>
<label>Horizontal zoom <input type="range" min="100" max="600" value="100" id="zoom"></label>
<div style="overflow:auto;width:100%"><div id="chart">'''+output.getvalue()+'''</div></div>
<script>const chart=document.querySelector('#chart svg');chart.style.height='auto';
document.querySelector('#zoom').oninput=e=>{chart.style.width=e.target.value+'%';};</script>'''
    (folder/'interactive_observed_PD_timeline.html').write_text(html)
    write(ROOT/'analysis/timeline_observations.json',dict(runs=evidence,evidence='TRACE_ANALYSIS',
        physical_GPU_idle='NOT_MEASURED',stream_identity='Observer current stream. No complete per-kernel multistream capture.',
        request_selection='First actually migrated Revised request at or beyond the cohort midpoint. Use the same request ID in every arm. No latency selects this example.',
        scope='Per-run CUDA event enclosure coordinates with Host anchors. Host wait includes admission and transfer. The diagnostic failed clean tail qualification.'))
    print('DIAGNOSTIC_TIMELINE_READY')


if __name__=='__main__':main()
