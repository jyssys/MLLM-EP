"""Plot only retained measurements. Keep modeled sums separate."""
import csv,statistics
from .common import *


def main():
    initialize()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=ROOT/'figures';output.mkdir(exist_ok=True);manifest=[]
    def save(fig,name,scope,sources):
        fig.tight_layout();path=output/(name+'.png');fig.savefig(path,dpi=160);plt.close(fig)
        manifest.append(dict(file=str(path.relative_to(ROOT)),SHA256=digest(path),scope=scope,sources=sources))
    value=json.loads((ROOT/'analysis/fixed_policy_confirmation.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,model in zip(axes,('huginn','ouro')):
        rows=[r for r in value['width'] if r['model']==model]
        for start in (1,2,3):
            line=[r for r in rows if r['start']==start]
            ax.plot([r['token_batch'] for r in line],[r['token_over_loop'] for r in line],'-o',label=f'Start {start}')
        ax.axhline(1,color='black',ls='--',lw=1);ax.set_xscale('log',base=2)
        ax.set_xlabel('Compatible Token requests batched (not serving admission)')
        ax.set_ylabel('Token / Loop isolated dense-time sum');ax.set_title(model.capitalize());ax.legend()
    fig.suptitle('Fixed shape-specific GEMM policy — structural sums of measured projection times\nNot an executed recurrent core or serving speedup')
    save(fig,'fixed_kernel_width_and_batching','Measured isolated projection timings with structural multiplicities. Compatible work and KV admission are not supplied by this figure.',['analysis/fixed_policy_confirmation.json'])
    cost=ROOT/'analysis/serial_path_costs.csv'
    if cost.exists():
        rows=list(csv.DictReader(cost.open()));fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,model in zip(axes,('huginn','ouro')):
            selected=[r for r in rows if r['model']==model and 'zero' not in r['name'] and
                (r['name'].startswith('cost_v2_') if model=='huginn' else r['name'].startswith('cost_ouro_'))]
            labels=[r['name'].replace('cost_v2_huginn_','').replace('cost_'+model+'_','').replace('_serial8_s1','').replace('_serial8_s2','') for r in selected]
            ax.bar(range(len(selected)),[float(r['D_sum_p50_ms']) for r in selected],color=['#197a82' if r['policy'].startswith('tuned') else '#a7a7a7' for r in selected])
            ax.set_xticks(range(len(selected)),labels,rotation=65,ha='right');ax.set_ylabel('D prompt enclosure p50 (ms)');ax.set_title(model.capitalize())
        fig.suptitle('Actual serial Native-path continuation — original and fixed tuned kernels\nCUDA Event stream enclosures, not exclusive kernels or Decode-safe slack')
        save(fig,'serial_native_path_cost','Measured serial enclosures. Prelude, Coda, Attention, and host launch gaps can contribute.',['analysis/serial_path_costs.csv'])
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,model in zip(axes,('huginn','ouro')):
            for row in rows:
                if row['model']!=model or not row['policy'].startswith('tuned') or 'zero' in row['name']:continue
                if model=='huginn' and not row['name'].startswith('cost_v2_'):continue
                if model=='ouro' and not row['name'].startswith('cost_ouro_'):continue
                x=float(row['D_sum_p50_ms']);y=float(row['P_saved_sum_p50_ms'])
                label=row['name'].replace('cost_v2_huginn_','').replace('cost_'+model+'_','').replace('_tuned_serial8_s1','').replace('_tuned_serial8_s2','')
                ax.scatter(x,y,s=45);ax.annotate(label,(x,y),fontsize=8)
            ax.axhline(0,color='black',lw=.7);ax.set_title(model.capitalize());ax.set_xlabel('D prompt enclosure p50 (ms)');ax.set_ylabel('Paired P enclosure saving p50 (ms)')
        fig.suptitle('Serial redistribution economics after kernel tuning\nPaired enclosure differences are not additive E2E components')
        save(fig,'serial_redistribution_frontier','Measured P/D serial enclosure comparison. No actual idle-window or serving admission claim.',['analysis/serial_path_costs.csv'])
        core=[r for r in rows if r['name'].startswith('core_cost_') and r['D_recurrent_graph_sum_p50_ms']]
        if core:
            fig,ax=plt.subplots(figsize=(9,4))
            labels=[r['name'].replace('core_cost_huginn_','').replace('_serial8_s1','') for r in core]
            ax.bar(range(len(core)),[float(r['D_recurrent_graph_sum_p50_ms']) for r in core],
                color=['#197a82' if r['policy'].startswith('tuned') else '#a7a7a7' for r in core])
            ax.set_xticks(range(len(core)),labels,rotation=35,ha='right');ax.set_ylabel('Recurrent graph enclosure sum p50 (ms)')
            ax.set_title('Huginn actual legal recurrent core — Prelude/Coda excluded\nCUDA Events include graph output clones and launch gaps. Not exclusive kernels.')
            save(fig,'actual_recurrent_core_width','Measured packed recurrent graph enclosures in serial controls. No additional recurrent replay or omitted work.',['analysis/serial_path_costs.csv'])
    serving=ROOT/'analysis/serving_results.json'
    if serving.exists():
        rows=[r for r in json.loads(serving.read_text()) if r.get('complete')]
        groups=sorted({(r['model'],r['cap'],r['count']) for r in rows})
        for model,cap,count in groups:
            selected=[r for r in rows if (r['model'],r['cap'],r['count'])==(model,cap,count)]
            labels=[label for label in ('zero','loop','loop28','token','token_group32','token1792') if any(r['width_arm']==label for r in selected)]
            names=dict(zero='Native Zero',loop='Loop31' if model=='huginn' else 'Loop3',loop28='Loop28',
                token='Token64 paced' if model=='huginn' else 'Token1024',token_group32='Token64 grouped',token1792='Token256 paced')
            fig,axes=plt.subplots(2,3,figsize=(15,7))
            for ax,field in zip(axes.flat,('TTFT_p95_ms','E2E_p95_ms','SLO_goodput','ITL_p99_ms','TPOT_p95_ms','maximum_ITL_ms')):
                for index,label in enumerate(labels):
                    points=[r[field] for r in selected if r['width_arm']==label]
                    ax.scatter([index]*len(points),points,s=30);ax.plot([index-.2,index+.2],[statistics.mean(points)]*2,color='black')
                ax.set_xticks(range(len(labels)),[names[label] for label in labels],rotation=35,ha='right');ax.set_ylabel(field.replace('_',' '))
            fig.suptitle(f'{model.capitalize()} O{cap}, N{count} — marker-free finite-cohort serving\nEach point is an independent process start. Service TTFT is NOT_MEASURED.')
            save(fig,f'serving_{model}_O{cap}_N{count}','Measured coordinator delivery latency. Not sustainable capacity or physical GPU activity.',['analysis/serving_results.json'])
    write(output/'figure_manifest.json',manifest)
    print('FIGURES',len(manifest))


if __name__=='__main__':main()
