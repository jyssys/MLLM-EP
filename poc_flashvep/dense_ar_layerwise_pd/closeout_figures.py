"""Show the final family and actual diagnostic kernels with explicit scopes."""
import csv,json
from .common import *
from .final_figures import main as earlier_figures
from .physical_analysis import merge


def main():
    earlier_figures()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=ROOT/'figures';manifest=json.loads((out/'figure_manifest.json').read_text())
    def save(fig,name,evidence,sources,scope):
        fig.tight_layout();fig.savefig(out/(name+'.svg'));fig.savefig(out/(name+'.png'),dpi=150);plt.close(fig)
        manifest['figures'].append(dict(name=name,evidence=evidence,sources=sources,scope=scope))
    final=ROOT/'analysis/primed_N128_confirmation.json'
    if final.exists():
        data=json.loads(final.read_text());fig,axes=plt.subplots(1,4,figsize=(13,3.4))
        for ax,key in zip(axes,['TTFT_p95_ms','E2E_p95_ms','ITL_p99_ms','TPOT_p95_ms']):
            for x,label,color in [(0,'layer','#2caa8c'),(1,'colocated','#5186b4')]:
                z=data['paired_statistics'][label][key];point=z['paired_geometric_mean'];a,b=z['CI95']
                ax.errorbar(x,point,yerr=[[point-a],[b-point]],fmt='o',color=color,capsize=4)
                ax.scatter([x-.07+i*.07 for i in range(len(z['start_ratios']))],z['start_ratios'],s=12,color=color)
            ax.axhline(1,color='black',linewidth=.8)
            if key.startswith(('ITL','TPOT')):ax.axhline(1.05,color='#cc3333',linestyle='--')
            ax.set(xticks=[0,1],xticklabels=['Layer','Colocated'],ylabel='Ratio to primed Native',title=key.replace('_ms','').replace('_',' '));ax.grid(axis='y',alpha=.2)
        save(fig,'figure_7_final_text_enabled_confirmation','MEASURED_MARKER_FREE_PAIRED_STARTS',
            [str(final.relative_to(ROOT))],'Independent held-out arrival Starts. Three Start-unit bootstrap intervals have limited power. Text output remains enabled for all methods.')
    diag=ROOT/'analysis/migrated_kernel_timeline_diagnostic.json'
    if diag.exists():
        data=json.loads(diag.read_text());folder=ROOT/'raw'/data['name']
        w=json.loads((folder/'window.json').read_text());origin=w['start'];spans=[]
        for role in ('prefill','decode'):
            anchor=json.loads((folder/f'activity_{role}_begin.json').read_text())['anchor']
            offset=anchor['host_midpoint']-anchor['cupti_ns']/1e9
            kernels=[];copies=[]
            with (folder/f'activity_{role}.csv').open() as f:
                for row in csv.DictReader(f):
                    a=max(origin,int(row['start_ns'])/1e9+offset);b=min(w['end'],int(row['end_ns'])/1e9+offset)
                    if b<=a:continue
                    if row['kind']=='kernel':kernels.append((a-origin,b-origin))
                    elif row['kind'].startswith('memcpy'):copies.append((a-origin,b-origin))
            for kind,values in [('kernel',kernels),('memcpy',copies)]:
                spans.extend(dict(role=role,kind=kind,start=a,end=b) for a,b in merge(values))
            frames=json.loads((folder/f'CUDA_event_enclosures_{role}.json').read_text())['rows']
            for row in frames:
                if not all(r.startswith('q') and r[1:].isdigit() for r in row['requests']):continue
                spans.append(dict(role=role,kind='model_event',start=row['start_monotonic']-origin,
                    end=row['end_monotonic']-origin,requests=row['requests'],prefilling=row['prefilling']))
        fig,ax=plt.subplots(figsize=(12,4))
        lanes=[('prefill','kernel','P actual kernels','#e7a146'),('decode','kernel','D actual kernels','#5186b4'),
            ('decode','memcpy','D copies, including memcpy2','#aa77bb')]
        for i,(role,kind,label,color) in enumerate(lanes):
            selected=[r for r in spans if r['role']==role and r['kind']==kind]
            ax.broken_barh([(r['start'],r['end']-r['start']) for r in selected],(i-.2,.4),facecolors=color)
        ax.set(yticks=list(range(3)),yticklabels=[x[2] for x in lanes],xlabel='Seconds from common measured window',
            title='Actual migrated execution: diagnostic CUDA interval unions, not clean capacity')
        ax.grid(axis='x',alpha=.2)
        save(fig,'figure_8_actual_migration_kernels','DIAGNOSTIC_ACTUAL_CUDA_ACTIVITY',[str(diag.relative_to(ROOT))],
            'Full-capture execution only. Copy intervals include memcpy and memcpy2 records. Observer ratios and trace integrity are in the source JSON. These intervals do not certify clean Decode-safe capacity.')
        payload=json.dumps(dict(seconds=w['end']-origin,spans=spans,metadata=data['observer_ratios']),separators=(',',':'))
        html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Measured migration timeline</title>
<style>body{font:15px sans-serif;margin:24px}svg{border:1px solid #bbb;width:100%;height:340px}input{width:100px}p{max-width:1000px}</style>
<h1>Actual CUDA migration timeline</h1><p>DIAGNOSTIC capture. Kernel unions and memcpy use separate lanes. Hatched-outline model spans are CUDA-event enclosures, not exclusive kernel times. Blank time is not a certified Decode-safe opportunity. Hover over a span for its measured role, time, and requests.</p>
<label>Start seconds <input id="a" type="number" value="0" min="0" step="0.1"></label>
<label>End seconds <input id="b" type="number" step="0.1"></label><button onclick="draw()">Update window</button><svg id="plot" viewBox="0 0 1200 340"></svg><pre id="meta"></pre>
<script>const data=PAYLOAD;document.getElementById('b').value=Math.min(3,data.seconds);document.getElementById('meta').textContent='Instrumented / control ratios: '+JSON.stringify(data.metadata,null,2);
function draw(){const a=+document.getElementById('a').value,b=+document.getElementById('b').value,s=document.getElementById('plot');s.replaceChildren();if(!(b>a))return;
const lanes=['prefill:kernel','prefill:model_event','decode:kernel','decode:model_event','decode:memcpy'];const ns='http://www.w3.org/2000/svg';
for(let i=0;i<lanes.length;i++){let t=document.createElementNS(ns,'text');t.setAttribute('x',5);t.setAttribute('y',45+i*58);t.textContent=lanes[i];s.append(t)}
for(const r of data.spans){let y=lanes.indexOf(r.role+':'+r.kind);if(y<0||r.end<=a||r.start>=b)continue;let v=document.createElementNS(ns,'rect');v.setAttribute('x',160+1020*(Math.max(a,r.start)-a)/(b-a));v.setAttribute('width',Math.max(.3,1020*(Math.min(b,r.end)-Math.max(a,r.start))/(b-a)));v.setAttribute('y',25+y*58);v.setAttribute('height',25);v.setAttribute('fill',r.kind==='model_event'?'none':r.role==='prefill'?'#e7a146':r.kind==='kernel'?'#5186b4':'#aa77bb');v.setAttribute('stroke',r.kind==='model_event'?'#2caa8c':'none');let title=document.createElementNS(ns,'title');title.textContent=JSON.stringify(r);v.append(title);s.append(v)}}draw();</script></html>'''.replace('PAYLOAD',payload)
        (out/'actual_migration_timeline.html').write_text(html)
        manifest['figures'].append(dict(name='actual_migration_timeline.html',evidence='DIAGNOSTIC_ACTUAL_CUDA_ACTIVITY',
            sources=[str(diag.relative_to(ROOT))],scope='Interactive zoom of the actual instrumented execution. No clean idle or counterfactual speedup claim.'))
    write(out/'figure_manifest.json',manifest)
    print('CLOSEOUT_FIGURES',len(manifest['figures']))


if __name__=='__main__':main()
