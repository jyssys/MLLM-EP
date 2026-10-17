"""Plot final paired pilot evidence without changing earlier figures."""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_async_recurrent_pp.contract import *


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/final_plots')
    args=parser.parse_args()
    assert not args.output.exists(), 'Preserve earlier figures'
    args.output.mkdir(parents=True)
    rows=read(args.analysis/'canonical/paired_summary.json')['rows']
    arms=['exact','stale','delta','w0']
    for name,value,interval,ylabel in [
        ('answer_accuracy','answer_accuracy_percent','answer_accuracy_CI95','Answer accuracy, %'),
        ('completed_accuracy','accuracy_percent','accuracy_CI95','Completed-answer accuracy, %')]:
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for axis,dataset in zip(axes,['gsm8k','math500']):
            selected=[next(row for row in rows if row['dataset']==dataset and row['arm']==arm) for arm in arms]
            axis.bar(arms,[row[value] for row in selected])
            for i,row in enumerate(selected):
                axis.vlines(i,*row[interval],color='black')
            axis.set_title(f'{dataset}: pilot n={selected[0]["n"]}')
            axis.set_ylabel(ylabel)
            axis.set_ylim(0,100)
        fig.suptitle('Free-running greedy pilot. Question-bootstrap 95% CI.')
        fig.tight_layout()
        fig.savefig(args.output/(name+'.png'),dpi=160)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for axis,dataset in zip(axes,['gsm8k','math500']):
        for i,arm in enumerate(arms[1:]):
            row=next(row for row in rows if row['dataset']==dataset and row['arm']==arm)
            axis.plot(i,row['answer_delta_pp'],'o')
            axis.vlines(i,*row['answer_delta_pp_CI95'])
        axis.axhline(0,color='grey')
        axis.set_xticks(range(3),arms[1:])
        axis.set_title(dataset)
        axis.set_ylabel('Answer accuracy: candidate minus exact, pp')
    fig.suptitle('Paired question-bootstrap 95% CI. Pilot rejection, not full evaluation.')
    fig.tight_layout()
    fig.savefig(args.output/'paired_accuracy_delta.png',dpi=160)
    plt.close(fig)
    errors=read(ROOT/'outputs/diagnostics_v2/boundary_errors.json')
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for axis,arm in zip(axes,arms[1:]):
        for stage in (1,2,3):
            means=[np.mean([row['relative_L2'] for row in errors
                if row['arm']==arm and row['round']==r and row['stage']==stage]) for r in range(4)]
            axis.plot(range(1,5),means,marker='o',label=f'Stage {stage+1} input')
        axis.set_title(arm)
        axis.set_xlabel('Recurrent round')
        axis.set_ylabel('Mean effective relative L2 error')
        axis.set_xticks(range(1,5))
        axis.legend(fontsize=8)
    fig.suptitle('Teacher-forced calibration. Two requests, 32 positions each. Not task accuracy.')
    fig.tight_layout()
    fig.savefig(args.output/'effective_boundary_error.png',dpi=160)
    plt.close(fig)
    write(args.output/'scope.json',dict(source_analysis=str(args.analysis),
        task_figures='Measured free-running pilot accuracy. No full-benchmark or speedup claim.',
        boundary_figure='Calibration teacher forcing only. Token positions are not independent requests.',
        timeline='NOT_RUN: quality gate precedes distributed implementation.',
        speedup='NOT_RUN: no distributed timing or simulation was promoted.'))


if __name__=='__main__':
    main()
