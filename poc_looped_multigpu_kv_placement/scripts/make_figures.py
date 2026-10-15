"""Draw measured curves. Label logical accounts and diagnostic service explicitly."""
import csv
import collections
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_looped_multigpu_kv_placement.contract import *


def read(name):return json.loads((ROOT/'results'/name).read_text())
def csv_rows(name):return list(csv.DictReader((ROOT/'results'/name).open()))
def save(name):
    plt.tight_layout();plt.savefig(ROOT/'figures'/name,dpi=160,bbox_inches='tight');plt.close()


def main():
    (ROOT/'figures').mkdir(exist_ok=True);plt.rcParams.update({'font.size':9})
    native=read('native_summary.json')['groups'];prefixes=[1024,4096,8192,16384,32704]
    fig,axes=plt.subplots(1,3,figsize=(12,3.4))
    for ax,field,label in zip(axes,['engine_prefill_ms','engine_TPOT_ms','batch_wall_ms'],
            ['Native first-token envelope (ms)','Cached TPOT (ms/token)','Client E2E (ms)']):
        for tp in [1,2,4]:
            means=[native[f'TP{tp}_P{p}'][field]['mean'] for p in prefixes]
            ci=np.array([native[f'TP{tp}_P{p}'][field]['CI95'] for p in prefixes]).T
            error=np.maximum(0,np.array([np.array(means)-ci[0],ci[1]-np.array(means)]))
            ax.errorbar(np.array(prefixes)/1024,means,yerr=error,marker='o',label=f'TP{tp}')
        ax.set(xlabel='Prompt positions (Ki tokens)',ylabel=label);ax.grid(alpha=.25)
    axes[0].legend();fig.suptitle('Ouro R4: 8 requests × 3 repeats. Request-bootstrap 95% CI. 64 outputs.')
    save('native_b1_curves.png')
    owned=read('tp_ownership_proof.json');fig,ax=plt.subplots(figsize=(7,3.5));bottom=np.zeros(3)
    parts=[('measured_parameter_storage','Weights'),('KV_pool_storage_bytes','Allocated native KV pool'),('other_allocated','Other allocated')]
    for field,label in parts:
        values=[next(row[field] for row in owned if row['tp']==tp and row['rank']==0)/1024**3 for tp in [1,2,4]]
        ax.bar(['TP1','TP2','TP4'],values,bottom=bottom,label=label);bottom+=values
    reserved=[next(row['reserved'] for row in owned if row['tp']==tp and row['rank']==0)/1024**3 for tp in [1,2,4]]
    ax.scatter(range(3),reserved,color='black',marker='_',s=240,label='Reserved HBM')
    ax.set(ylabel='GiB per rank',title='Actual native ownership after graph capture. Pool is not request-filled KV.');ax.legend(fontsize=8)
    save('native_hbm_accounts.png')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    axes[0].bar(['TP1','TP2','TP4'],[1.5,.75,.375]);axes[0].set(ylabel='Logical MiB/token/rank',title='Ouro: 192 independent slots')
    hg=ROOT/'results/raw/huginn_default32_cache_probe_v1'
    names=['H16 full','H32 full','H16 compress-s16'];logical=[];storage=[]
    sources=[ROOT/'results/raw/huginn_smoke_v1/smoke_clean.json',hg/'smoke_clean.json',ROOT/'results/raw/huginn_smoke_v1/cache_policy_secondary.json']
    if not sources[0].exists():
        candidates=list((ROOT/'results/raw').glob('huginn*smoke*/smoke_clean.json'))
        if candidates:sources[0]=candidates[0];sources[2]=candidates[0].parent/'cache_policy_secondary.json'
    for path in sources:
        row=json.loads(path.read_text());logical.append(row['cache']['logical_unique_elements_bytes']/1024**2);storage.append(row['cache']['actual_unique_storage_bytes']/1024**2)
    x=np.arange(3);axes[1].bar(x-.18,logical,.36,label='Logical KV');axes[1].bar(x+.18,storage,.36,label='Unique backing')
    axes[1].set(xticks=x,xticklabels=names,ylabel='MiB for 35 cached positions',title='Huginn official cache. Compression changes outputs.');axes[1].legend()
    save('cache_geometry.png')
    batches=csv_rows('capacity_frontier.csv');replicas=csv_rows('replica_throughput.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,p in zip(axes,[1024,32704]):
        groups=collections.defaultdict(list)
        for row in batches:
            if int(row['prefix'])==p:groups[(f'TP{row["tp"]}',int(row['concurrency']))].append(float(row['aggregate_TPS']))
        for row in replicas:
            if int(row['prefix'])==p:groups[(f'TP{row["tp"]} × DP{row["dp"]}',int(row['concurrency']))].append(float(row['aggregate_TPS']))
        for name in sorted({key[0] for key in groups}):
            cs=sorted(c for n,c in groups if n==name);ax.plot(cs,[np.mean(groups[(name,c)]) for c in cs],marker='o',label=name)
        ax.set(xlabel='Offered concurrency, not resident count',ylabel='Actual finite-cohort output TPS',title=f'Prompt {p} + 64 outputs');ax.grid(alpha=.25);ax.legend(fontsize=7)
    save('four_gpu_native_throughput.png')
    service=read('kernel_summary.json')
    if service:
        categories=list(service[0]['category_service_ms']);fig,ax=plt.subplots(figsize=(9,3.5));bottom=np.zeros(len(service))
        for cat in categories:
            vals=[r['category_service_ms'].get(cat,0) for r in service];ax.bar(range(len(service)),vals,bottom=bottom,label=cat);bottom+=vals
        ax.set(xticks=range(len(service)),xlabel='Logical TP rank (physical 4–7)',ylabel='Accumulated kernel service (ms)',
            title='Diagnostic: 16K prefill + 3 cached forwards. Not an E2E percentage.');ax.legend(fontsize=6,bbox_to_anchor=(1.02,1))
        save('kernel_service_pressure.png')
    loops=csv_rows('loop_metrics.csv')
    if loops:
        cats=['QKV_PROJECTION','ROPE','ATTENTION_AND_KV_FUSED','O_PROJECTION_INCLUDES_REDUCE','MLP_INCLUDES_REDUCE','TP_ALL_REDUCE_API_INTERVAL_NESTED']
        matrix=np.zeros((4,len(cats)))
        for row in loops:
            if row['rank']=='0' and row['query_rows']=='1':matrix[int(row['loop']),cats.index(row['category'])]+=float(row['service_or_interval_ms'])/3
        fig,ax=plt.subplots(figsize=(10,3));im=ax.imshow(matrix,aspect='auto');ax.set(yticks=range(4),ylabel='Recurrent pass',xticks=range(len(cats)),xticklabels=cats,title='UNVALIDATED fine timing: eager observer +24.6%. Nested reduction is not additive.')
        plt.setp(ax.get_xticklabels(),rotation=20,ha='right');fig.colorbar(im,label='ms, diagnostic only');save('loop_component_heatmap.png')
    fig,ax=plt.subplots(figsize=(10,2.7));ax.axis('off')
    rows=[['Native TP4','W/4 per rank','K/4 per rank','384 model reductions/forward','Measured native'],
        ['TP2 + CP2 KV','W/2 on two ranks','K/4 ideal; more on KV-only ranks if rebalanced','Extra Q / partial O / LSE / fresh-KV traffic','Latency UNKNOWN'],
        ['One loop per rank','Shared core copied four times','K/4 ideal','3 hidden-state boundaries + token feedback','B1 loops stay sequential']]
    wrapped=[['\n'.join(textwrap.wrap(value,29)) for value in row] for row in rows]
    ledger=ax.table(cellText=wrapped,colLabels=['Layout','Weight ownership','KV ownership','Communication','Status'],
        loc='center',cellLoc='left',colWidths=[.14,.19,.23,.27,.17])
    ledger.auto_set_font_size(False);ledger.set_fontsize(7.4);ledger.scale(1,2.8)
    ax.set_title('Conceptual placement ledger. No candidate speedup was measured.');save('placement_tradeoff_ledger.png')
    fig,ax=plt.subplots(figsize=(10,3));ax.axis('off')
    quality=[['Ouro R4','Math','416, EOS','Correct arithmetic answer'],
        ['Ouro R4','Coding','512, capped','No completed function within cap'],
        ['Ouro R4','Document','512, capped','Incomplete, unscored'],
        ['Huginn H16','Math','153, EOS','Wrong arithmetic answer'],
        ['Huginn H16','Coding','232, EOS','Requested function absent'],
        ['Huginn H16','Document','58, EOS','Factual accuracy unscored']]
    ax.table(cellText=quality,colLabels=['Model/regime','Calibration case','Output length / finish','Descriptive assessment'],
        loc='center',cellLoc='left',colWidths=[.15,.15,.2,.5])
    ax.set_title('Natural-EOS sanity, 3 cases/model. Not benchmark accuracy. No candidate runtime.');save('quality_sanity.png')
    print('Wrote measured curves and labelled memory/service figures.')


if __name__=='__main__':main()
