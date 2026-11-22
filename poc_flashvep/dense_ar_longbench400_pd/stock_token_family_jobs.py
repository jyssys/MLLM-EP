"""Validate a dense strong-token cut family with the original compiled model."""
from .experiment import create
from .freeze_capsule import freeze
from .next_jobs import plan


def main():
    capsule,sha=freeze('stock_token_cut_family_qualification_v1_20261121')
    lanes=[[],[]]
    for index,cut in enumerate((1,2,4,12)):
        lane=index%2;first=(0,4)[lane]
        base=dict(pair=(first,first+1),correctness=True,execution_family='INSTALLED_DEFAULT_COMPILE',
            source_capsule=capsule,source_capsule_SHA256=sha,
            gpu_state_digest=True,wall_timeout=1800,serving_timeout=1500)
        jobs=[create(f'Q13_StockFull36_Teacher{cut}_N8_v1','Native','qualification8.json',
            output=16,fixed=True,serial=True,reference_single=True,
            token_teacher_cut=cut,state_capture=True,axis_state_capture=False,
            numerical_probe=True,numerical_probe_sync=True,kv_layout='HND',**base),
            create(f'Q13_StockToken{cut}_N8_FullState_v1','Token','qualification8.json',cut=cut,
                output=16,fixed=True,serial=True,controlled=True,token_stock_model=True,
                axis_state_capture=True,axis_capture_hidden_last_only=True,axis_import_hash_after_model=True,**base),
            create(f'Q13_StockToken{cut}_C3_Natural32_v1','Token','continuous_qualification32_natural.json',cut=cut,
                token_stock_model=True,qps=8,seed=2026112311,
                axis_state_capture=True,axis_capture_hidden_last_only=True,axis_import_hash_after_model=True,
                continuous_source_credit_limit=0,**base)]
        lanes[lane].extend(jobs)
    for lane,jobs in enumerate(lanes):plan(f'stock_token_cut_family_qualification_lane{lane}_v1',jobs)
    print('Frozen original-model teachers and stock-token N8/N32 gates for cuts1,2,4,12.',flush=True)


if __name__=='__main__':main()
