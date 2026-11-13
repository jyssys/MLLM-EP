"""Plot frozen-model decisions and realized completion residuals."""
from .common import *


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=json.loads((ROOT/'analysis/revised_prediction_validation.json').read_text())['results']
    selected=[r for r in results if r['name'].startswith('heldout_Revised_O16_N128_q4.8_')]
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    colors={8:'#168d87',12:'#a86491',4:'#bd9834',2:'#4974a5'}
    observations=[p for r in selected for p in r['accepted_predictions']]
    for layers,color in colors.items():
        points=[r for r in observations if r['layers']==layers]
        if not points:continue
        axes[0].scatter([r.get('execution_budget_ms',r['idle_budget_ms']) for r in points],
            [layers]*len(points),color=color,alpha=.6,label=f'Last{layers}')
    errors=[1000*(r['observed_first_token']-r['predicted_D_finish_mean']) for r in observations]
    upper=[1000*(r['observed_first_token']-r['predicted_D_finish']) for r in observations]
    axes[1].hist(errors,bins=15,color='#5487b6',label='Central error')
    axes[1].axvline(0,color='black',ls=':');axes[1].legend()
    axes[2].scatter(errors,upper,alpha=.7,color='#188d89')
    axes[2].axhline(0,color='#b43c43',ls='--',label='Upper-estimate exceedance boundary')
    axes[2].legend(fontsize=8)
    axes[0].set(xlabel='Predicted execution budget (ms), NOT physical idle',ylabel='Actual committed layers',yticks=[0,2,4,8,12])
    axes[0].legend()
    axes[1].set(xlabel='Observed minus central completion prediction (ms)',ylabel='Committed groups')
    axes[2].set(xlabel='Central prediction error (ms)',ylabel='Observed minus upper estimate (ms)')
    fig.suptitle(f'Fresh Output16 / QPS4.8, three independent starts, {len(observations)} realized groups\nOnly actual committed migrations appear. Rejected candidates have no observed counterfactual.')
    fig.tight_layout();folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    for suffix in ('png','svg'):fig.savefig(folder/('figure_J_budget_and_prediction_error.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig);print('PREDICTION_FIGURE_READY',len(observations))


if __name__=='__main__':main()
