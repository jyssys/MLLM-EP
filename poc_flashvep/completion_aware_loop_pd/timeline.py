"""Render actual host observations without inventing GPU execution intervals."""
import json
from .common import *
from .audit_old import events


def run_data(root,name,label):
    folder=root/'raw'/name;begin=json.loads((folder/'measurement_begin.json').read_text())['time']
    reqs=[r for r in map(json.loads,(folder/'requests.jsonl').read_text().splitlines()) if not r.get('warmup')]
    ev=events(folder,begin)
    metrics={m['request_id']:m for m in json.loads((folder/'stats.json').read_text())['metrics']}
    spans=[r for p in folder.glob('atomic_spans_*.json') for v in [json.loads(p.read_text())]
        if v['role']=='prefill' for r in v['prompt_spans'] if r['time']>=begin]
    result=[]
    for r in reqs:
        m=metrics[f"r0_q{r['sample_id']}"];tid=m['transfer_id'];ts=m['timings'];origin=m['created_monotonic']
        p=sorted((s for s in spans if s['seed']==r['seed']),key=lambda s:s['time']);source=[]
        for index,s in enumerate(p):
            boundary=s['depth']+s['loop_count']
            ready=[e['time'] for e in ev if e.get('tid')==tid and e['event'] in (
                'GPU_COMPLETED_BOUNDARY_OFFER','SOURCE_LOCAL_BOUNDARY_SKIP') and e.get('depth')==boundary and e['time']>=s['time']]
            stop=min(ready) if ready else p[index+1]['time'] if index+1<len(p) else origin+ts['prefill_compute_completed']
            source.append(dict(start=s['time'],end=max(s['time'],stop),
                label=f"T{s['start']}–{s['start']+s['count']} / L{s['depth']+1}–{boundary}",kind='P host enclosure',rows=s['count']))
        quanta=[dict(start=e['submitted_host_time'],end=e['time'],label=f"{e['rows']} rows × {e['loops']} loops",
            kind='D host enclosure') for e in ev if e['event']=='D_QUANTUM_HOST_COMPLETION' and
            (e.get('choice') or {}).get('tid')==tid]
        commit=next((e['time'] for e in ev if e['event']=='SOURCE_COMMIT' and e.get('tid')==tid),None)
        received=origin+ts['d_received'] if 'd_received' in ts else None
        handoff_begin=commit if commit is not None else (
            origin+ts['prefill_compute_completed'] if 'prefill_compute_completed' in ts else None)
        committed=next((e for e in ev if e['event']=='SOURCE_COMMIT' and e.get('tid')==tid),{})
        result.append(dict(index=r['sample_id'],seed=r['seed'],arrival=r['arrival'],first=r['first_token'],
            finish=r['finish'],ttft_ms=r['ttft_ms'],p=source,d=quanta,commit=commit,received=received,
            handoff_begin=handoff_begin,
            offloaded_loops=committed.get('remaining',0) if not committed.get('token_split') else 0,
            offloaded_suffix_tokens=committed.get('remaining',0) if committed.get('token_split') else 0))
    return dict(label=label,name=name,requests=result,decode_outputs=[e['time'] for r in reqs for e in r['token_events'] if not e['first']],
        scope='Host enclosures and output-delivery events only. Blank space does not identify physical GPU idle.')


HTML='''<!doctype html><meta charset="utf-8"><title>Native / old / new observed timeline</title>
<style>body{font:15px system-ui;margin:24px;color:#253044;background:#fafbfc}h1{font-size:23px}select,input{margin:8px}svg{background:white;border:1px solid #ddd;width:100%}.note{max-width:1200px;line-height:1.5}text{font-family:system-ui}</style>
<h1>Native / old / new: observed request timeline</h1>
<p class="note">These are independent serving runs. Each row starts at the selected request's arrival. Bars are host enclosures, not GPU kernels. Blank space is unobserved GPU activity. Blue marks identify output delivery, not Decode kernel duration.</p>
<label>Model <select id="model"><option>ouro</option><option>huginn</option></select></label>
<label>Request index <select id="request"></select></label><label>View duration (s) <input id="duration" type="number" min="0.1" step="0.1"></label>
<svg id="plot" viewBox="0 0 1400 610"></svg><p id="summary"></p>
<p class="note">Orange: P submission-to-ready host enclosure. Green: D submission-to-observed-completion host enclosure. Purple: source commit-to-D receipt host interval. Native uses its observed Prefill-complete-to-D-receipt interval. Purple does not measure DMA alone. Red: first output. Shared batch enclosures can overlap. Do not add their durations. The initial view selects the first observed offloaded request, not the fastest request. All 128 requests remain selectable.</p>
<script>const DATA=__DATA__;
const model=document.getElementById('model'),request=document.getElementById('request'),duration=document.getElementById('duration'),plot=document.getElementById('plot');
function options(){request.replaceChildren();let arms=DATA[model.value];for(let i=0;i<128;i++){let o=document.createElement('option');o.value=i;let marked=arms.some(a=>a.requests.some(r=>r.index===i&&(r.offloaded_loops>0||r.offloaded_suffix_tokens>0)));o.textContent=i+(marked?' · actual offload':'');request.append(o)}let preferred=arms.slice().reverse().find(a=>a.label.toLowerCase().includes('loop')&&a.requests.some(r=>r.offloaded_loops>0));let first=preferred?preferred.requests.find(r=>r.offloaded_loops>0):arms.flatMap(a=>a.requests).find(r=>r.offloaded_loops>0||r.offloaded_suffix_tokens>0);request.value=first?first.index:0}
function element(tag,attributes,text){let n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(let k in attributes)n.setAttribute(k,attributes[k]);if(text!==undefined)n.textContent=text;plot.append(n);return n}
function draw(reset){plot.replaceChildren();let arms=DATA[model.value],id=Number(request.value);plot.setAttribute('viewBox','0 0 1400 '+(100+170*arms.length));let selected=arms.map(a=>a.requests.find(r=>r.index===id));if(reset)duration.value=Math.max(.5,...selected.map(r=>r?(r.first-r.arrival)*1.12:0)).toFixed(2);let span=Number(duration.value)||1;const left=170,width=1200,x=t=>left+width*t/span;
element('text',{x:170,y:25,'font-size':15,fill:'#a33b4a'},'HOST OBSERVATIONS ONLY · no measured GPU idle or kernel bars');
for(let i=0;i<arms.length;i++){let a=arms[i],r=selected[i],top=65+i*170;if(!r)continue;element('text',{x:8,y:top+10,'font-size':16},a.label);element('text',{x:8,y:top+31,'font-size':12},'P host');element('text',{x:8,y:top+76,'font-size':12},'D continuation');element('text',{x:8,y:top+121,'font-size':12},'Output delivery');
function bar(s,e,y,color,label){s=Math.max(0,s-r.arrival);e=Math.min(span,e-r.arrival);if(e<s||s>span)return;let box=element('rect',{x:x(s),y:y,width:Math.max(2,x(e)-x(s)),height:27,fill:color,opacity:.8});let title=document.createElementNS('http://www.w3.org/2000/svg','title');title.textContent=label+' | '+s.toFixed(6)+'–'+e.toFixed(6)+' s';box.append(title);if(x(e)-x(s)>65)element('text',{x:x(s)+4,y:y+18,'font-size':11,fill:'white'},label)}
for(let b of r.p)bar(b.start,b.end,top+15,'#dd9d45',b.label);for(let b of r.d)bar(b.start,b.end,top+60,'#148979',b.label);if(r.handoff_begin!==null&&r.received!==null)bar(r.handoff_begin,r.received,top+93,'#8b6eac','Handoff host interval');
for(let t of a.decode_outputs){t-=r.arrival;if(t>=0&&t<=span)element('line',{x1:x(t),x2:x(t),y1:top+108,y2:top+127,stroke:'#4386bd','stroke-width':1,opacity:.6})}
let first=r.first-r.arrival;if(first<=span){element('line',{x1:x(first),x2:x(first),y1:top,y2:top+132,stroke:'#ca4352','stroke-width':2});element('text',{x:x(first)+3,y:top+9,'font-size':12,fill:'#b93446'},'first output '+first.toFixed(3)+' s')}
element('text',{x:170,y:top+150,'font-size':12},a.name+' | full-prompt loops: '+r.offloaded_loops+' | suffix tokens: '+r.offloaded_suffix_tokens+' | TTFT: '+r.ttft_ms.toFixed(2)+' ms')}
let axis=55+170*arms.length;for(let i=0;i<=8;i++){let t=span*i/8;element('line',{x1:x(t),x2:x(t),y1:axis,y2:axis+8,stroke:'#666'});element('text',{x:x(t)-10,y:axis+26,'font-size':12},t.toFixed(2))}element('text',{x:650,y:axis+43,'font-size':12},'Time from selected request arrival (s)');document.getElementById('summary').textContent='Same request index and paired arrival seed. Each band uses its own measured clock. Use the final report for cohort-level results.'}
model.onchange=()=>{options();draw(true)};request.onchange=()=>draw(true);duration.onchange=()=>draw(false);options();draw(true);
</script>'''


def main():
    data={}
    for model in ('ouro','huginn'):
        data[model]=[run_data(ROOT,f'primary_{model}_{arm}_seed2026110911',label)
            for arm,label in (('native','Native'),('old','Old Loop'),('loop','New Loop'))]
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    write(folder/'observed_timeline_data.json',data)
    (folder/'native_old_new_observed_timeline.html').write_text(HTML.replace('__DATA__',json.dumps(data)),encoding='utf-8')
    manifest=json.loads((ROOT/'figure_manifest.json').read_text())
    item=dict(name='native_old_new_observed_timeline',html='figures/native_old_new_observed_timeline.html',
        scope='Same primary arrival seed. Host observations only. Blank space is not physical GPU idle.',
        sources=['figures/observed_timeline_data.json'])
    write(ROOT/'figure_manifest.json',[r for r in manifest if r['name']!=item['name']]+[item])
    print(folder/'native_old_new_observed_timeline.html')


if __name__=='__main__':main()
