"""Tune token chunks in Serving, not only in isolated P service cohorts."""
from .common import *
from .search import serving_config


def main():
    references=json.loads((ROOT/'configs/capacity_references.json').read_text());new=[]
    for model,length,chunks,caps in [('ouro',8192,(256,1024,2048),(32,64,100,200)),
        ('huginn',3900,(256,1024),(32,64,100))]:
        cell=references[model+'_L'+str(length)]
        for cap in caps:
            mu=min(max(c['mu_P'] for c in cell['chunks'].values()),cell['D'][str(cap)]['mu_D'])
            for chunk in chunks:
                for factor in ((.75,1.,1.25) if cap==32 else (1.,)):
                    name=f'pilot_{model}_l{length}_o{cap}_serving_chunk{chunk}_f{str(factor).replace(".","p")}_v1'
                    c=serving_config(cell,name,cap,chunk,'qps_pilot',24,rate=mu*factor)
                    c.update(rate_factor=factor,reference_capacity=mu,
                        tuning_scope='Equal three-rate chunk screen at output 32. Other output caps use a one-rate diagnostic. Isolated P-optimal C is not assumed to be Serving-optimal.')
                    new.append(save_config(c))
    path=ROOT/'configs/study_plan.json';old=json.loads(path.read_text())['configs']
    write(path,dict(configs=list(dict.fromkeys(new+old))))
    write(ROOT/'state/serving_chunk_tuning_plan.json',dict(configs=new,
        reason='Native transport and cooperative progress can change the optimal Serving chunk. Compare all qualified C settings instead of selecting C only from independent P rates.'))
    print('SERVING_CHUNK_CONTROLS',len(new))


if __name__=='__main__':main()
