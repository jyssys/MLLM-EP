"""Run the declared BI0 reference before selecting performance candidates."""
from .common import ROOT, write, digest, save, initialize
from .dataset import main as dataset
from .run import execute


def main():
    initialize()
    dataset()
    cfg = dict(name='BI0_Standard_ShortAnswer200_v2_Natural_C1_cap2048_v1', method='Native',
        entry_module='poc_flashvep.dense_ar_bi0_shortanswer_pd.natural_reference',
        gpus=[1],batch_invariant=False,kv_layout='HND',token_budget=8192,max_num_seqs=1024,
        context_limit=40960,output_cap=2048,dataset='longbench_shortanswer200_v2.json',
        dataset_SHA256=digest(ROOT/'data/longbench_shortanswer200_v2.json'),
        execution_family='STOCK_BI0',wall_timeout=3600,seed=2026112001,phase='PHASE_A_NATURAL_REFERENCE')
    assert execute(save(cfg)), cfg['name']
    write(ROOT/'state/phase_A_reference_complete.json',dict(complete=True,name=cfg['name']))


if __name__ == '__main__':
    main()
