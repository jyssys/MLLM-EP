"""Plot only the measured C1 lengths and retain every plotted value as CSV."""
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .common import ROOT,write,digest
from .c0_analysis import csv_save

def main():
    manifest=json.loads((ROOT/'data/balanced400_v2_freeze_manifest.json').read_text())
    rows=json.loads((ROOT/'data/longbench_balanced400_v2.json').read_text());folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    fields=['prompt_id','task','category','input_tokens','natural_output_tokens','input_bucket','natural_output_bucket','evaluation_split','context_SHA256','input_ids_SHA256']
    with (folder/'v2_dataset_input_output.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
    with (folder/'v2_dataset_cell_counts.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['input_bucket','output_bucket','available_original_candidates','selected','target'])
        for i in range(4):
            for j in range(4):writer.writerow([i,j,manifest['available_cells'][i][j],manifest['selected_cells'][i][j],25])
    note='Measured original-prompt C1 natural EOS lengths | BI0 Mode0 prompt/native decode graph | n=400 | cap2048,0 selected cap hits\nNot C3 output lengths or a production distribution. Source revision5e628be; dataset SHA256 '+manifest['SHA256'][:16]
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    histograms=[]
    for index,(field,bins,color) in enumerate((('input_tokens',30,'#4477aa'),('natural_output_tokens',25,'#228833'))):
        counts,edges=np.histogram([r[field] for r in rows],bins=bins)
        ax[index].hist([r[field] for r in rows],bins=edges,color=color)
        histograms += [dict(series=field,bin_index=j,left_edge=edges[j],right_edge=edges[j+1],
            right_edge_inclusive=j==len(counts)-1,count=int(count),n=len(rows)) for j,count in enumerate(counts)]
    ax[0].set(xlabel='Input tokens after chat template',ylabel='Requests',title='LongBench400-v2 input distribution')
    ax[1].set(xlabel='Generated output tokens',ylabel='Requests',title='C1 natural output distribution')
    csv_save(folder/'v2_dataset_length_distribution.csv',histograms)
    fig.text(.01,.01,note,fontsize=8);fig.tight_layout(rect=[0,.12,1,1])
    for ext in ('png','pdf'):fig.savefig(folder/f'v2_dataset_length_distribution.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for category in sorted({r['category'] for r in rows}):
        part=[r for r in rows if r['category']==category]
        ax[0].scatter([r['input_tokens'] for r in part],[r['natural_output_tokens'] for r in part],s=15,alpha=.65,label=category)
    ax[0].set(xlabel='Input tokens',ylabel='C1 natural output tokens',title='Original-prompt joint distribution');ax[0].legend(fontsize=7)
    counts=np.asarray(manifest['selected_cells']);im=ax[1].imshow(counts,cmap='Blues',vmin=0,vmax=max(25,int(counts.max())))
    ax[1].set_xticks(range(4),['1–10','11–50','51–100','101–200']);ax[1].set_yticks(range(4),['3K–6K','6K–10K','10K–20K','20K–36K'])
    ax[1].set(xlabel='C1 output-token bin',ylabel='Input-token bin',title='Selected counts; target25/cell')
    for i in range(4):
        for j in range(4):ax[1].text(j,i,str(counts[i,j]),ha='center',va='center',color='white' if counts[i,j]>35 else 'black')
    plotted=[dict(record_type='joint_point',**{k:r[k] for k in fields}) for r in rows]
    plotted += [dict(record_type='selected_cell',input_bucket=i,natural_output_bucket=j,
        selected_count=int(counts[i,j]),target_count=25) for i in range(4) for j in range(4)]
    csv_save(folder/'v2_dataset_joint_and_balance.csv',plotted)
    fig.colorbar(im,ax=ax[1],label='Requests');fig.text(.01,.01,note,fontsize=8);fig.tight_layout(rect=[0,.12,1,1])
    for ext in ('png','pdf'):fig.savefig(folder/f'v2_dataset_joint_and_balance.{ext}',dpi=180)
    plt.close(fig)
    files={p.name:digest(p) for p in folder.glob('v2_dataset_*')}
    write(ROOT/'analysis/dataset_figure_manifest_v2.json',dict(files=files,dataset_SHA256=manifest['SHA256'],evidence='MEASURED_NATURAL_REFERENCE'))
    print('DATASET_FIGURES_PNG_PDF_CSV_COMPLETE',len(files),flush=True)

if __name__=='__main__':main()
