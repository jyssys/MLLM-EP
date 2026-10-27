"""Publication values verified against the primary figure and paper tables."""
import csv
from .common import ROOT,setup,digest,DATA,write

def main():
    setup();records=[]
    blog='https://huskydoge.github.io/husky-blog/posts/recursive_models/towards-looped-models-done-right/'
    values=dict(arc_challenge=(51.33,50.47),hellaswag=(54.82,55.10),mmlu=(64.03,66.81),
        bbh_cot=(69.05,68.85),tqa=(50.98,51.04),drop=(57.70,59.65),math=(49.80,53.40),
        gsm8k=(78.92,83.62),human_eval_plus=(48.17,49.39),mbpp_plus=(58.99,59.52))
    for task,pair in values.items():
        for model,score in zip(['ifm_ouro','ifm_huginn'],pair):
            records.append(dict(model=model,task=task,reference_value=score,unit='PERCENT',table='Primary MoE benchmark figure, 500B tokens',
                url=blog,reference_scope='OFFICIAL_PUBLISHED_SCORE',standard_error=None))
    paper='https://arxiv.org/html/2609.35751v1'
    for task,score,se in [('prost',29.45,.33),('lambada_standard',31.83,.65),('lambada_openai',39.98,.68),
        ('swag',59.98,.35),('hellaswag',48.43,.50),('blimp',81.82,.13),('xwinograd_en',68.82,.96)]:
        records.append(dict(model='foil1',task=task,reference_value=score,unit='PERCENT',table='Table 5, 100B, 0-shot',url=paper,
            reference_scope='OFFICIAL_PUBLISHED_SCORE',standard_error=se))
    for task,value in [('fineweb_edu_heldout',2.4492),('wikipedia_heldout',2.5311),('c4_heldout',2.9584),('arxiv_heldout',2.7084)]:
        records.append(dict(model='foil1',task=task,reference_value=value,unit='NATS_PER_TOKEN',table='Table 7, 100B',url=paper,
            reference_scope='OFFICIAL_PUBLISHED_HELDOUT_NLL',standard_error=None))
    path=ROOT/'accuracy_reference.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
    write(ROOT/'quality/reference_evidence.json',dict(ifm_figure_sha256=digest(DATA/'ifm_table_moe.png'),
        ifm_figure_url=blog+'table-moe-benchmarks.png',foil_paper=paper,
        excluded_training_loss=2.4988,excluded_reason='Training final-window loss is not a held-out target.'))
    print('REFERENCE_ROWS',len(records))

if __name__=='__main__':main()
