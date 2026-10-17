"""Render measured calibration diagnostics and paired pilot statistics."""
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_async_recurrent_pp.contract import *


def heatmap(matrix,name,title,label):
    fig,axis=plt.subplots(figsize=(6,4))
    picture=axis.imshow(matrix,aspect='auto')
    axis.set_xticks(range(3),['Stage 2','Stage 3','Stage 4'])
    axis.set_yticks(range(3),['R2','R3','R4'])
    axis.set_title(title)
    for i in range(3):
        for j in range(3):
            axis.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white')
    fig.colorbar(picture,ax=axis,label=label)
    fig.tight_layout()
    fig.savefig(ROOT/'outputs/plots'/name,dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wave',type=Path)
    args=parser.parse_args()
    directory=ROOT/'outputs/plots'
    directory.mkdir(exist_ok=True)
    observations=read(ROOT/'outputs/baseline_v1/activation_observations.json')
    for field,name,label in [('adjacent_cosine','loop_cosine.png','Cosine similarity'),
                              ('stale_relative_L2','loop_relative_L2.png','Relative L2')]:
        matrix=np.array([[np.mean([row[field] for row in observations if row['round']==r and row['stage']==j])
                          for j in range(1,4)] for r in range(1,4)])
        heatmap(matrix,name,'Exact calibration: 8 requests, 136 positions per cell',label)
    diagnostics=ROOT/'outputs/diagnostics_v2/teacher_metrics.json'
    if diagnostics.exists():
        records=read(diagnostics)
        labels=sorted({row['arm'] for row in records})
        fig,axis=plt.subplots(figsize=(10,5))
        means=[np.mean([row['KL_exact_candidate_nats'] for row in records if row['arm']==arm]) for arm in labels]
        axis.bar(range(len(labels)),means)
        axis.set_xticks(range(len(labels)),labels,rotation=35,ha='right')
        axis.set_ylabel('Mean KL(exact || candidate), nats')
        axis.set_title('Teacher-forced calibration injections: 2 requests, 64 positions per arm')
        fig.tight_layout();fig.savefig(directory/'teacher_KL.png',dpi=160);plt.close(fig)
    if args.wave:
        rows=read(args.wave/'paired_summary.json')['rows']
        arms=['exact','stale','delta','w0']
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for axis,dataset in zip(axes,['gsm8k','math500']):
            selected=[next(row for row in rows if row['dataset']==dataset and row['arm']==arm) for arm in arms]
            axis.bar(arms,[row['accuracy_percent'] for row in selected])
            for i,row in enumerate(selected):
                lo,hi=row['accuracy_CI95']
                axis.vlines(i,lo,hi,color='black')
            axis.set_title(f'{dataset}: pilot n={selected[0]["n"]}')
            axis.set_ylabel('Completed accuracy, %')
            axis.set_ylim(0,100)
        fig.suptitle('Free-running greedy evaluation; prompt bootstrap 95% CI')
        fig.tight_layout();fig.savefig(directory/'pilot_accuracy.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for axis,dataset in zip(axes,['gsm8k','math500']):
            selected=[next(row for row in rows if row['dataset']==dataset and row['arm']==arm) for arm in arms[1:]]
            for i,row in enumerate(selected):
                axis.plot(i,row['delta_pp'],'o')
                axis.vlines(i,*row['delta_pp_CI95'])
            axis.axhline(0,color='grey')
            axis.set_xticks(range(3),arms[1:])
            axis.set_title(dataset)
            axis.set_ylabel('Candidate minus exact, percentage points')
        fig.suptitle('Paired prompt-bootstrap 95% CI; pilot only')
        fig.tight_layout();fig.savefig(directory/'paired_accuracy_delta.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    main()
